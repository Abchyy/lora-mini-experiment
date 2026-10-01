#!/usr/bin/env python3
"""
Minimal LoRA fine-tune: DistilBERT + GLUE SST-2 (sequence classification).

Follows the ~2h recipe in LoRA技术调研报告-v2-2026-10-01.md §5,
scaled for CPU-only boxes (smaller subset / batch / seq length).

No paid APIs. Uses Hugging Face Transformers + PEFT + datasets + evaluate.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal DistilBERT LoRA on SST-2")
    p.add_argument("--model_id", default="distilbert-base-uncased")
    p.add_argument("--max_train_samples", type=int, default=800)
    p.add_argument("--max_eval_samples", type=int, default=200)
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_lin", "v_lin"],
        help="DistilBERT attention linear names (not in PEFT default map)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Trainer checkpoints dir (default: <project>/outputs/run)",
    )
    p.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
        help="Where to save PEFT adapter (default: <project>/results/adapter)",
    )
    p.add_argument(
        "--metrics_path",
        type=str,
        default=None,
        help="JSON metrics path (default: <project>/results/metrics.json)",
    )
    p.add_argument("--dataloader_num_workers", type=int, default=0)
    return p.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def hardware_summary() -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / (1024**3), 2)
        info["ram_available_gb"] = round(vm.available / (1024**3), 2)
    except Exception:
        info["ram_total_gb"] = None
        info["ram_available_gb"] = None
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    else:
        info["gpu_name"] = None
        info["gpu_count"] = 0
    return info


def accuracy_metric(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


def main() -> None:
    args = parse_args()
    root = project_root()
    output_dir = Path(args.output_dir or root / "outputs" / "run")
    adapter_dir = Path(args.adapter_dir or root / "results" / "adapter")
    metrics_path = Path(args.metrics_path or root / "results" / "metrics.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    hw = hardware_summary()
    print("=== Hardware ===")
    print(json.dumps(hw, indent=2))
    print("=== Config ===")
    print(json.dumps(vars(args), indent=2, default=str))

    t0 = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, num_labels=2
    )

    # DistilBERT is NOT in PEFT TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
    # (verified in research report for PEFT v0.21.0). Must set target_modules explicitly.
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(args.target_modules),
        bias="none",
        modules_to_save=["classifier", "pre_classifier"],
        use_rslora=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    raw = load_dataset("glue", "sst2")
    n_train = min(args.max_train_samples, len(raw["train"]))
    n_eval = min(args.max_eval_samples, len(raw["validation"]))
    raw["train"] = raw["train"].shuffle(seed=args.seed).select(range(n_train))
    raw["validation"] = (
        raw["validation"].shuffle(seed=args.seed).select(range(n_eval))
    )

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = raw.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    keep = {"input_ids", "attention_mask", "labels"}
    tokenized = tokenized.remove_columns(
        [c for c in tokenized["train"].column_names if c not in keep]
    )

    # Prefer simple numpy accuracy to avoid optional glue metric download quirks.
    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to=[],
        seed=args.seed,
        fp16=use_fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        remove_unused_columns=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=accuracy_metric,
    )

    print("=== Training ===")
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    print("=== Eval metrics ===")
    print(eval_metrics)

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    elapsed_s = time.perf_counter() - t0
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")

    payload = {
        "experiment": "distilbert-sst2-lora-mini",
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": round(elapsed_s, 2),
        "hardware": hw,
        "sources": {
            "model": args.model_id,
            "dataset": "glue/sst2 (Hugging Face datasets)",
            "frameworks": {
                "transformers": __import__("transformers").__version__,
                "peft": __import__("peft").__version__,
                "datasets": __import__("datasets").__version__,
                "torch": torch.__version__,
            },
            "recipe": "LoRA技术调研报告-v2-2026-10-01.md §5 (CPU-scaled)",
        },
        "config": {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "scaling": args.lora_alpha / args.lora_r,
            "use_rslora": False,
            "lora_dropout": args.lora_dropout,
            "target_modules": list(args.target_modules),
            "bias": "none",
            "modules_to_save": ["classifier", "pre_classifier"],
            "task_type": "SEQ_CLS",
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "max_length": args.max_length,
            "max_train_samples": n_train,
            "max_eval_samples": n_eval,
            "seed": args.seed,
        },
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": round(100.0 * trainable / total, 4) if total else None,
        "train_metrics": {
            k: (float(v) if hasattr(v, "item") else v)
            for k, v in train_result.metrics.items()
        },
        "eval_metrics": {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in eval_metrics.items()
        },
        "paths": {
            "adapter_dir": str(adapter_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "metrics_path": str(metrics_path.resolve()),
        },
        "interpretation": {
            "process_verification": True,
            "strong_conclusions": False,
            "note": (
                "This run verifies the LoRA+PEFT training pipeline on a tiny SST-2 "
                "subset under CPU constraints. Accuracy is not comparable to Hu et al. "
                "Table 2 (RoBERTa/DeBERTa full SST-2) and must not be cited as a "
                "benchmark claim."
            ),
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote metrics -> {metrics_path}")
    print(f"Saved adapter -> {adapter_dir}")
    print(f"Elapsed: {elapsed_s:.1f}s")


if __name__ == "__main__":
    main()
