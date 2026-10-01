# Minimal LoRA Experiment (DistilBERT + SST-2)

ZGCA / 创智学院 prep: end-to-end **PEFT LoRA** fine-tune on a tiny GLUE SST-2 subset.
Recipe follows `LoRA技术调研报告-v2-2026-10-01.md` §5, **CPU-scaled** (fewer samples, shorter sequence, 2 epochs).

This verifies the **training process** (freeze base → train A/B + classifier → save adapter → log metrics). It is **not** a strong benchmark claim vs Hu et al. Table 2.

**Repo:** https://github.com/Abchyy/lora-mini-experiment

## Layout

```
lora-mini-experiment/
  requirements.txt
  README.md
  EXPERIMENT_REPORT.md
  .gitignore
  src/train_lora_sst2.py
  scripts/run_experiment.sh
  logs/                 # sample run log (tracked); new runs append more
  results/              # metrics.json + adapter config/tokenizer (tracked)
  outputs/              # Trainer checkpoints (gitignored; created on run)
  hf_cache/             # local HF cache (gitignored; created on first run)
  .venv/                # local virtualenv (gitignored; created on setup)
```

## What is in git vs what you regenerate

### Tracked in this repository (already uploaded)

| Path | Notes |
|------|--------|
| `src/`, `scripts/`, `requirements.txt`, `.gitignore` | Code and deps |
| `README.md`, `EXPERIMENT_REPORT.md` | Docs |
| `results/metrics.json`, `results/run.log` | Metrics and a copy of the sample run stdout |
| `logs/run-20261001-120905.log` | Full sample run log |
| `results/adapter/adapter_config.json` | LoRA config metadata |
| `results/adapter/tokenizer*`, `vocab.txt`, `special_tokens_map.json`, `README.md` | Tokenizer side-car files |

### Not in git — create by following Reproduce below

| Artifact | Why missing |
|----------|-------------|
| `.venv/` | Local environment |
| `hf_cache/` | Downloaded DistilBERT + SST-2 |
| `outputs/` | Trainer checkpoints |
| `results/adapter/adapter_model.safetensors` | ~3 MB LoRA weights (ignored via `*.safetensors`) |
| New `logs/run-*.log` / `logs/latest.log` | Only produced when you re-run |

Re-running the script **overwrites** `results/metrics.json` and regenerates the adapter (including weights). The committed sample metrics still match the original CPU run described in `EXPERIMENT_REPORT.md`.

## Hardware note

Designed for **CPU-only**. If a GPU is present, the script enables `fp16` automatically.

## Reproduce (from a fresh clone)

```bash
git clone https://github.com/Abchyy/lora-mini-experiment.git
cd lora-mini-experiment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
bash scripts/run_experiment.sh
```

`scripts/run_experiment.sh` resolves the repo root from its own location, so it does **not** depend on a fixed host path like `/workspace/...`. If `.venv` is missing, the script creates it and installs `requirements.txt` before training.

One-liner after the first setup (venv already present):

```bash
bash scripts/run_experiment.sh
```

Optional env overrides (CPU defaults shown):

```bash
MAX_TRAIN_SAMPLES=800 MAX_EVAL_SAMPLES=200 NUM_EPOCHS=2 BATCH_SIZE=8 \
  bash scripts/run_experiment.sh
```

Closer to the report’s §5 recipe (expect longer wall-clock on CPU; **not re-verified** in this repo’s sample run):

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

## Outputs after a successful run

- `logs/run-*.log` — full stdout/stderr (`logs/latest.log` symlink on Unix)
- `results/metrics.json` — config, hardware, train/eval metrics
- `results/adapter/` — PEFT adapter (`adapter_config.json` + `adapter_model.safetensors` + tokenizer files)

## Command / path verification status

| Item | Status |
|------|--------|
| Default CPU run via `bash scripts/run_experiment.sh` (800/200, 2 epochs, batch 8) | **Verified** on the original experiment host (2026-10-01); see `results/metrics.json` and `EXPERIMENT_REPORT.md` |
| Script path resolution relative to repo root (no hard-coded `/workspace`) | **Verified** in `scripts/run_experiment.sh` |
| Fresh `git clone` → venv → `pip install` → run on another machine/OS | **Not re-verified** here; expected to work if Python 3.10+ and network to Hugging Face Hub are available |
| GPU / `fp16` path | **Not verified** in the sample run (host was CPU-only) |
| Larger §5-style overrides (2000/500, 3 epochs, …) | **Not verified** in the sample run |
| Windows PowerShell activation / bash availability | **Not verified** |

## License / sources

- Model & tokenizer: Hugging Face `distilbert-base-uncased`
- Dataset: GLUE SST-2 via `datasets.load_dataset("glue", "sst2")`
- LoRA API: Hugging Face PEFT (`LoraConfig`, `get_peft_model`)
- No paid APIs
