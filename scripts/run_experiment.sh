#!/usr/bin/env bash
# End-to-end LoRA mini experiment runner (CPU-friendly defaults).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${ROOT}/.venv"
PYTHON="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

if [[ ! -x "$PYTHON" ]]; then
  echo "[setup] Creating venv at ${VENV}"
  python3 -m venv "$VENV"
  "$PIP" install --upgrade pip setuptools wheel
  "$PIP" install -r "${ROOT}/requirements.txt"
fi

export HF_HOME="${HF_HOME:-${ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false
# Prefer CPU when no GPU; avoid CUDA init surprises on CPU-only hosts.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

mkdir -p "${ROOT}/logs" "${ROOT}/results" "${ROOT}/outputs"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="${ROOT}/logs/run-${STAMP}.log"
LATEST_LOG="${ROOT}/logs/latest.log"

echo "[run] Logging to ${LOG}"
set +e
"$PYTHON" -u "${ROOT}/src/train_lora_sst2.py" \
  --max_train_samples "${MAX_TRAIN_SAMPLES:-800}" \
  --max_eval_samples "${MAX_EVAL_SAMPLES:-200}" \
  --max_length "${MAX_LENGTH:-64}" \
  --num_train_epochs "${NUM_EPOCHS:-2}" \
  --per_device_train_batch_size "${BATCH_SIZE:-8}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-16}" \
  --learning_rate "${LR:-5e-4}" \
  --lora_r "${LORA_R:-8}" \
  --lora_alpha "${LORA_ALPHA:-8}" \
  --lora_dropout "${LORA_DROPOUT:-0.1}" \
  "$@" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e
ln -sfn "$(basename "$LOG")" "$LATEST_LOG"
exit "$RC"
