# LoRA Mini Experiment Report

**Project:** [Abchyy/lora-mini-experiment](https://github.com/Abchyy/lora-mini-experiment) (paths below are relative to the repo root)  
**Date:** 2026-10-01 (Asia/Shanghai, UTC+8)  
**Recipe source:** `LoRA技术调研报告-v2-2026-10-01.md` §5 (CPU-scaled)  
**Published:** https://github.com/Abchyy/lora-mini-experiment (public; sample run committed)

---

## 1. Goal

Implement and run a **minimal PEFT LoRA** fine-tune to verify the end-to-end process for ZGCA / 创智学院 prep:

1. Freeze DistilBERT base weights  
2. Attach low-rank adapters on attention Q/V  
3. Train adapters + classification head on a tiny SST-2 subset  
4. Save adapter checkpoint and logged metrics  

**This is process verification, not a strong empirical conclusion.**

---

## 2. Hardware (measured this run)

| Item | Value |
|------|-------|
| Device | **CPU only** (`nvidia-smi` unavailable; `torch.cuda.is_available() == False`) |
| CPU | 8× Intel Xeon (KVM), x86_64 |
| RAM | 15.64 GiB total; ~3.8–4.5 GiB available at start |
| Disk | Experiment host workspace overlay ~118 GiB free at run time |
| Python | 3.13.5 |
| Torch | 2.8.0+cu128 (CPU execution) |
| Wall-clock (train+eval+save) | **25.71 s** |

---

## 3. Sources

| Component | Source |
|-----------|--------|
| Base model | Hugging Face `distilbert-base-uncased` |
| Tokenizer | same checkpoint |
| Dataset | GLUE SST-2 via `datasets.load_dataset("glue", "sst2")` |
| LoRA API | PEFT `LoraConfig` + `get_peft_model` + `TaskType.SEQ_CLS` |
| Trainer | Hugging Face Transformers `Trainer` |
| Paid APIs | **None** |

Framework versions (this run): transformers `4.57.6`, peft `0.17.1`, datasets `3.6.0`, accelerate `1.15.0`, torch `2.8.0+cu128`.

---

## 4. Config (CPU defaults actually run)

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| `r` | 8 | Hu Table 9 / PEFT default |
| `lora_alpha` | 8 | scaling = α/r = **1.0** |
| `use_rslora` | False | standard LoRA scaling |
| `lora_dropout` | 0.1 | |
| `target_modules` | `["q_lin", "v_lin"]` | DistilBERT names; **not** in PEFT default map — must set explicitly |
| `bias` | `"none"` | |
| `modules_to_save` | `["classifier", "pre_classifier"]` | DistilBERT head stack |
| `task_type` | `SEQ_CLS` | |
| Learning rate | `5e-4` | Hu Table 9 RoBERTa-base SST-2 LoRA LR |
| Epochs | 2 | CPU scale-down (report §5 suggests 3) |
| Train batch | 8 | CPU scale-down (report: 16) |
| Eval batch | 16 | |
| `max_length` | 64 | CPU scale-down (report: 128) |
| Train samples | 800 | CPU scale-down (report: 2000) |
| Eval samples | 200 | CPU scale-down (report: 500) |
| Seed | 42 | |

### Trainable parameter count (measured)

- Trainable: **739,586**  
- Total: **67,694,596**  
- Trainable %: **1.0925%**  

Breakdown (derived): LoRA on 6×{q_lin,v_lin} ≈ 147,456; DistilBERT `pre_classifier` (768×768+bias) + `classifier` (768×2+bias) ≈ 592,130. LoRA-only fraction of the base is much smaller; the head dominates trainable count on this SEQ_CLS setup.

---

## 5. Metrics (this run)

From `results/metrics.json` (also mirrored in the committed sample log):

| Metric | Value |
|--------|-------|
| `train_loss` | 0.4048 |
| `train_runtime` (s) | 23.32 |
| `train_samples_per_second` | 68.62 |
| `eval_loss` | 0.3893 |
| `eval_accuracy` | **0.845** (200-example subset) |
| `eval_runtime` (s) | 0.87 |
| Elapsed wall-clock (s) | 25.71 |

Log: `logs/run-20261001-120905.log` (symlink `logs/latest.log` is local-only after a run)  
Adapter config/tokenizer: `results/adapter/`  
Adapter weights: regenerate `results/adapter/adapter_model.safetensors` with `bash scripts/run_experiment.sh` (not stored in git)

---

## 6. Interpretation — process verification vs strong conclusions

### Process verification (what this run *does* support)

- PEFT LoRA attaches to DistilBERT when `target_modules=["q_lin","v_lin"]` are set explicitly.  
- Base weights stay frozen; only LoRA factors + head train (~1.1% of params).  
- Training loop completes on CPU; loss decreases; adapter + metrics are saved.  
- Repo-relative runner in `scripts/run_experiment.sh` works from the checkout root.

### Not strong conclusions (do **not** claim)

- **Do not** compare `0.845` to Hu et al. Table 2 SST-2 numbers (RoBERTa/DeBERTa, full data, different setups).  
- Subset size (800/200), short seq (64), 2 epochs → **toy** accuracy only.  
- No ablation on `r`, α, target modules, or full vs LoRA fine-tune.  
- No latency/merge benchmark; merge API not exercised in this minimal run.

---

## 7. Limitations

1. **CPU-only** host for the sample run; no GPU timing.  
2. **Tiny subset** — not GLUE leaderboard protocol.  
3. **Head-heavy trainable count** — DistilBERT `pre_classifier` is large relative to LoRA A/B.  
4. First install pulled CUDA wheels for torch even though execution is CPU (disk/time cost).  
5. `accelerate` had to be `>=1.3` for Transformers 4.57 `unwrap_model(..., keep_torch_compile=...)`.  
6. No paid APIs, but first run needs Hugging Face Hub network access for model/dataset.  
7. Adapter **weights** are gitignored; clones must re-run to obtain `.safetensors`.

---

## 8. How to reproduce

Start from a clone (do not assume any host-specific path such as `/workspace/...`):

```bash
git clone https://github.com/Abchyy/lora-mini-experiment.git
cd lora-mini-experiment
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
bash scripts/run_experiment.sh
```

One-liner if the venv already exists inside the clone:

```bash
bash scripts/run_experiment.sh
```

### Verification status for these commands

| Step | Status |
|------|--------|
| Default training on the original experiment host | **Verified** (metrics above) |
| Path-independent `ROOT` resolution in `scripts/run_experiment.sh` | **Verified** by script design / that host run |
| End-to-end `git clone` on a clean second machine | **Not re-verified** after publishing |
| GPU path / larger env overrides | **Not verified** in the sample run |

See also the table in `README.md` → *Command / path verification status*.

---

## 9. Exam-style questions (3)

**Q1.** DistilBERT is not in PEFT’s default LoRA `target_modules` mapping. If you omit `target_modules` when calling `LoraConfig` for DistilBERT, what happens, and which module names should you use to mirror Hu et al.’s \(W_q, W_v\) choice?

*Answer sketch:* PEFT raises an error when the architecture is unknown / unmapped (or fails to match). DistilBERT attention linears are named `q_lin`, `k_lin`, `v_lin`, `out_lin`; use `["q_lin", "v_lin"]` for the paper’s Q/V pattern.

**Q2.** With \(r=8\), \(\alpha=8\), `use_rslora=False`, what is the forward scaling applied to \(BA x\)? How would that scaling change if `use_rslora=True` with the same \(r,\alpha\)?

*Answer sketch:* Standard: \(\alpha/r = 1\). rsLoRA: \(\alpha/\sqrt{r} = 8/\sqrt{8} = 2\sqrt{2} \approx 2.828\). Same \(\alpha,r\) ⇒ rsLoRA update is \(\sqrt{r}\) times larger.

**Q3.** Why initialize \(B=0\) (and \(A\) random / Kaiming) at the start of LoRA training? What does that imply for \(\Delta W\) at step 0?

*Answer sketch:* So \(\Delta W = BA = 0\) at initialization — the adapted model matches the frozen pretrained forward pass before any gradient step (identity / no-op adapter). Paper: Gaussian \(A\), zero \(B\); microsoft/PEFT Linear default: Kaiming \(A\), zero \(B\).

---

## 10. Deliverables checklist

| Path (repo-relative) | Role | In git? |
|----------------------|------|---------|
| `README.md` | Reproduce instructions | Yes |
| `EXPERIMENT_REPORT.md` | This report | Yes |
| `requirements.txt` | Dependencies | Yes |
| `src/train_lora_sst2.py` | Training entrypoint | Yes |
| `scripts/run_experiment.sh` | End-to-end runner | Yes |
| `results/metrics.json` | Logged metrics (sample run) | Yes |
| `results/run.log` | Sample run stdout copy | Yes |
| `logs/run-20261001-120905.log` | Sample run log | Yes |
| `results/adapter/adapter_config.json` + tokenizer files | Adapter metadata | Yes |
| `results/adapter/adapter_model.safetensors` | Adapter weights | **No** — regenerate |
| `.venv/`, `hf_cache/`, `outputs/` | Env / cache / checkpoints | **No** — regenerate |
| `.gitignore` | Ignores venv/caches/weights | Yes |

**Git status:** Public repository published at https://github.com/Abchyy/lora-mini-experiment (initial sample run pushed; docs updated in follow-up commits).
