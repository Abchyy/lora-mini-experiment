# Minimal LoRA Experiment (DistilBERT + SST-2)

ZGCA / 创智学院 prep: end-to-end **PEFT LoRA** fine-tune on a tiny GLUE SST-2 subset.
Recipe follows `LoRA技术调研报告-v2-2026-10-01.md` §5, **CPU-scaled** (fewer samples, shorter sequence, 2 epochs).

This verifies the **training process** (freeze base → train A/B + classifier → save adapter → log metrics). It is **not** a strong benchmark claim vs Hu et al. Table 2.

## Layout

```
lora-mini-experiment/
  requirements.txt
  README.md
  EXPERIMENT_REPORT.md
  .gitignore
  src/train_lora_sst2.py
  scripts/run_experiment.sh
  logs/                 # stdout/stderr captures
  results/              # metrics.json + adapter/
  outputs/              # Trainer checkpoints
  hf_cache/             # local HF cache (created on first run)
  .venv/                # local virtualenv
```

## Hardware note

Designed for **CPU-only**. If a GPU is present, the script enables `fp16` automatically.

## Reproduce (fresh terminal)

```bash
cd /workspace/lora-mini-experiment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
bash scripts/run_experiment.sh
```

One-liner (venv already created):

```bash
cd /workspace/lora-mini-experiment && bash scripts/run_experiment.sh
```

Optional env overrides (CPU defaults shown):

```bash
MAX_TRAIN_SAMPLES=800 MAX_EVAL_SAMPLES=200 NUM_EPOCHS=2 BATCH_SIZE=8 \
  bash scripts/run_experiment.sh
```

Closer to the report’s §5 recipe (expect longer wall-clock on CPU):

```bash
MAX_TRAIN_SAMPLES=2000 MAX_EVAL_SAMPLES=500 NUM_EPOCHS=3 BATCH_SIZE=16 MAX_LENGTH=128 \
  bash scripts/run_experiment.sh
```

## Config (CPU default run)

| Item | Value |
|------|-------|
| Model | `distilbert-base-uncased` |
| Data | GLUE `sst2` (HF `datasets`) |
| Subset | 800 train / 200 eval |
| Epochs | 2 |
| Batch | 8 |
| LR | `5e-4` |
| `r` / `alpha` | 8 / 8 → scaling `α/r = 1.0` |
| `target_modules` | `q_lin`, `v_lin` (DistilBERT; **must** set explicitly) |
| `modules_to_save` | `classifier`, `pre_classifier` |
| `use_rslora` | `False` |

## Outputs

- `logs/run-*.log` — full stdout/stderr (`logs/latest.log` symlink)
- `results/metrics.json` — config, hardware, train/eval metrics
- `results/adapter/` — PEFT adapter (`adapter_config.json` + weights)

## License / sources

- Model & tokenizer: Hugging Face `distilbert-base-uncased`
- Dataset: GLUE SST-2 via `datasets.load_dataset("glue", "sst2")`
- LoRA API: Hugging Face PEFT (`LoraConfig`, `get_peft_model`)
- No paid APIs
