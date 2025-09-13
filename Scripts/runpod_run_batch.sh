#!/bin/bash
set -euo pipefail

WORKDIR="/workspace/thesis"
VENV="$WORKDIR/.venv"

cd "$WORKDIR"

# Tokens to fill in:
HF_TOK="token"
OPENAI_TOK="token"
GROQ_TOK="token"

export HF_TOKEN="$HF_TOK"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export OPENAI_API_KEY="$OPENAI_TOK"
export GROQ_API_KEY="$GROQ_TOK"

if [[ ! -d "$VENV" ]]; then
  echo "[ERROR] venv not found at $VENV. Create it first:"
  echo "  python3 -m venv $VENV && source $VENV/bin/activate && pip install -r ..."
  exit 1
fi
source "$VENV/bin/activate"

export PYTHONPATH="$WORKDIR/SRC:${PYTHONPATH:-}"
cd "$WORKDIR/SRC/Modular_Learning_Agent"

RUN_ID=$(date +%Y%m%d_%H%M%S)
EVAL_DIR="$WORKDIR/Evaluation/sgd_eval_$RUN_ID"
mkdir -p "$EVAL_DIR"
export EVAL_RUN_DIR="$EVAL_DIR"
python -m Modular_Learning_Agent.main_batch