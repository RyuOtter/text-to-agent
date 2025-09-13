#!/bin/bash
set -e
VENV_PATH="/workspace/thesis/.venv"
python3.11 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
pip install -U "pip==25.2" "setuptools==75.8.0" "wheel==0.45.1"
pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" "triton==3.1.0"
  
cat > /tmp/constraints_updated.txt << 'EOF'
accelerate==1.10.0
bitsandbytes==0.46.1
datasets==3.6.0
diffusers==0.34.0
huggingface-hub==0.34.4
peft==0.17.0
transformers==4.49.0
tokenizers==0.21.4
sentencepiece==0.2.0
xformers==0.0.28.post3
cut-cross-entropy==25.1.1
pandas==2.3.1
tqdm==4.67.1
pydantic==2.11.7
pydantic_core==2.33.2
safetensors==0.6.1
numpy==1.26.4
protobuf==5.29.5
pyarrow==21.0.0
requests==2.32.4
urllib3==2.5.0
rich==14.1.0
tyro==0.9.28
termcolor==2.5.0
openai==1.99.9
python-dotenv==1.0.1
google-generativeai==0.8.5
groq==0.31.0
tenacity==8.5.0
matplotlib==3.10.5
click==8.2.1
PyYAML==6.0.2
langchain_community==0.3.13
EOF

pip install -c /tmp/constraints_updated.txt \
  accelerate transformers peft datasets bitsandbytes xformers \
  tokenizers sentencepiece diffusers pandas tqdm cut-cross-entropy \
  termcolor openai python-dotenv google-generativeai groq tenacity \
  matplotlib click PyYAML rich tyro langchain_community
pip install git+https://github.com/huggingface/trl.git@fc4dae256d924dfbb906af9c2e817bc6fb7b590b
pip install --no-deps "unsloth==2025.8.6" "unsloth_zoo==2025.8.5"
pip install flash-attn --no-build-isolation | cat || true
pip install vllm==0.7.3 | cat || true
pip install deepspeed==0.17.4 | cat || true
pip install liger-kernel==0.6.1 | cat || true
pip install wandb==0.21.1 | cat || true
pip install -U "huggingface_hub[cli]" | cat || true
pip install hf_transfer | cat || true
mkdir -p "$VENV_PATH/../SRC"
if [[ ! -d "$VENV_PATH/../SRC/Multi-Turn-RL-Agent" ]]; then
  git clone --depth=1 https://github.com/SiliangZeng/Multi-Turn-RL-Agent "$VENV_PATH/../SRC/Multi-Turn-RL-Agent" | cat
else
fi