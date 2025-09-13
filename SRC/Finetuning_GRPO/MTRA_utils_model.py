from __future__ import annotations
import torch
import verifiers as vf

# Load model in MTRA style based on Multi-Turn-RL-Agent folder
def load_mtra_model(model_name, model_kwargs=None):
    default_kwargs = {"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2", "use_cache": False}
    
    if model_kwargs:
        allowed = {"torch_dtype", "attn_implementation", "use_cache", "quantization_config"}
        safe_kwargs = {k: v for k, v in model_kwargs.items() if k in allowed}
        default_kwargs.update(safe_kwargs)
    
    return vf.get_model_and_tokenizer(model_name, model_kwargs=default_kwargs)