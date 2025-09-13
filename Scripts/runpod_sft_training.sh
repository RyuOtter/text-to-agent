#!/bin/bash
set -euo pipefail

WORKDIR="/workspace/thesis"
VENV="$WORKDIR/.venv"

FAST_CACHE_ROOT="${FAST_CACHE_ROOT:-/workspace/thesis/.hf_cache}"
TMPDIR="/workspace/thesis/tmp"
CHECKPOINT_DIR="/workspace/thesis/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

mkdir -p "$FAST_CACHE_ROOT"/hf_home "$FAST_CACHE_ROOT"/hub "$FAST_CACHE_ROOT"/transformers \
         "$FAST_CACHE_ROOT"/xdg "$FAST_CACHE_ROOT"/torch "$FAST_CACHE_ROOT"/triton \
         "$TMPDIR" "$CHECKPOINT_DIR"
cd "$WORKDIR"

# Tokens to fill in:
HF_TOK="token"
OPENAI_TOK="token"
GROQ_TOK="token"
WANDB_TOK="token"

export HF_TOKEN="$HF_TOK"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export OPENAI_API_KEY="$OPENAI_TOK"
export GROQ_API_KEY="$GROQ_TOK"
export WANDB_API_KEY="$WANDB_TOK"

export HF_HOME="$FAST_CACHE_ROOT/hf_home"
export HUGGINGFACE_HUB_CACHE="$FAST_CACHE_ROOT/hub"
export TRANSFORMERS_CACHE="$FAST_CACHE_ROOT/transformers"
export XDG_CACHE_HOME="$FAST_CACHE_ROOT/xdg"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TORCH_HOME="$FAST_CACHE_ROOT/torch"
export TRITON_CACHE_DIR="$FAST_CACHE_ROOT/triton"

export TMPDIR="$TMPDIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -d "$VENV" ]]; then
  echo "[ERROR] venv not found at $VENV. Create it first:"
  echo "  python3 -m venv $VENV && source $VENV/bin/activate && pip install -r ..."
  exit 1
fi
source "$VENV/bin/activate"

if command -v hf >/dev/null 2>&1; then
  if ! hf auth whoami >/dev/null 2>&1; then
    printf "%s" "$HF_TOKEN" | hf auth login --stdin >/dev/null 2>&1
  fi
else
  if ! huggingface-cli whoami >/dev/null 2>&1; then
    huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1
  fi
fi

export PYTHONPATH="$WORKDIR/SRC:${PYTHONPATH:-}"
export PYTHONPATH="/workspace/thesis/SRC/Multi-Turn-RL-Agent:${PYTHONPATH}"
cd "$WORKDIR"
python3 SRC/Finetuning_GRPO/SFT_training_loop.py \
    --config SRC/Finetuning_GRPO/MTRA_config.yaml \
    --dataset Data/SGD_SFT_Data.jsonl