from __future__ import annotations
import os, datetime, json, shutil, torch
from transformers import TrainerCallback, TrainerState, TrainerControl
from peft import get_peft_model_state_dict
from safetensors.torch import save_file
from huggingface_hub import create_repo, upload_folder
import verifiers as vf
TRANSFORMERS_AVAILABLE = True


# Save LoRA adapters during training as checkpoints
class LoRACheckpointCallback(TrainerCallback):
    
    def on_save(self, args, state, control, model=None, tokenizer=None, **kwargs):          
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            if hasattr(model, "peft_config") and model.peft_config:
                model.save_pretrained(checkpoint_dir)
            elif hasattr(model, "save_adapter"):
                model.save_adapter(checkpoint_dir)
            else:
                try:
                    adapter_state_dict = get_peft_model_state_dict(model)
                    adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
                    save_file(adapter_state_dict, adapter_path)
                except Exception as fallback_e:
                    return
            
            if tokenizer is not None:
                tokenizer.save_pretrained(checkpoint_dir)
                
        except Exception as e:
            return


# Create result and checkpoint directories
def setup_directories(log_cfg):
    
    run_stamp_format = log_cfg.get("run_stamp_format", "%Y%m%d-%H%M%S")
    run_stamp = datetime.datetime.now().strftime(run_stamp_format)
    checkpoints_root = log_cfg.get("checkpoints_root", "/workspace/thesis/checkpoints")
    results_root = log_cfg.get("results_dir_root", "/workspace/thesis/Results")
    run_checkpoint_root = os.path.join(checkpoints_root, f"mtra-{run_stamp}")
    run_results_dir = os.path.join(results_root, run_stamp)
    
    os.makedirs(run_checkpoint_root, exist_ok=True)
    os.makedirs(run_results_dir, exist_ok=True)
    
    return run_checkpoint_root, run_results_dir, run_stamp


# Configure GRPO training arguments
def setup_training_args(grpo_cfg, log_cfg, model_name, train_dialogs, max_seq_len, wandb_enabled=False, run_results_dir=None):
    
    training_args = vf.get_default_grpo_config(
        run_name=log_cfg.get("wandb", {}).get("run_name", f"sgd-grpo-{model_name.split('/')[-1].lower()}"),
        num_gpus=int(grpo_cfg.get("num_gpus", 1)),
    )
    
    if run_results_dir:
        checkpoints_dir = os.path.join(run_results_dir, "checkpoints")
        training_args.output_dir = checkpoints_dir
        os.makedirs(checkpoints_dir, exist_ok=True)
    
    training_args.max_prompt_length = max_seq_len
    training_args.max_completion_length = int(grpo_cfg.get("max_tokens", 256))
    training_args.vllm_max_model_len = max_seq_len + int(grpo_cfg.get("max_tokens", 256))

    training_args.bf16 = bool(grpo_cfg.get("bf16", True))
    if not training_args.bf16 and bool(grpo_cfg.get("fp16", False)):
        training_args.fp16 = True
            
    training_args.gradient_checkpointing = bool(grpo_cfg.get("gradient_checkpointing", True))
    training_args.dataloader_pin_memory = bool(grpo_cfg.get("dataloader_pin_memory", False))
    training_args.dataloader_num_workers = int(grpo_cfg.get("dataloader_num_workers", 0))
    training_args.learning_rate = float(grpo_cfg.get("learning_rate", 1e-6))
    training_args.num_generations = int(grpo_cfg.get("group_size", 4))
    training_args.per_device_train_batch_size = int(grpo_cfg.get("per_device_train_batch_size", 2))
    training_args.gradient_accumulation_steps = int(grpo_cfg.get("gradient_accumulation_steps", 4))
    training_args.max_steps = int(grpo_cfg.get("num_epochs", 1)) * len(train_dialogs)
    training_args.beta = float(grpo_cfg.get("beta", 0.0))
    training_args.loss_type = grpo_cfg.get("loss_type", "grpo")
    training_args.max_grad_norm = float(grpo_cfg.get("max_grad_norm", 1.0))
    training_args.lr_scheduler_type = grpo_cfg.get("lr_scheduler_type", "constant")
    
    opt_cfg = grpo_cfg.get("optimizer", {})
    training_args.optim = opt_cfg.get("name", "adamw_torch")
    training_args.weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    adam_betas = opt_cfg.get("betas", [0.9, 0.999])
    training_args.adam_beta1 = float(adam_betas[0]) if len(adam_betas) > 0 else 0.9
    training_args.adam_beta2 = float(adam_betas[1]) if len(adam_betas) > 1 else 0.999
    training_args.adam_epsilon = float(opt_cfg.get("eps", 1e-8))
    
    samples_per_step = training_args.per_device_train_batch_size * training_args.num_generations
    total_dialogs = len(train_dialogs)
    if samples_per_step > total_dialogs:
        training_args.per_device_train_batch_size = max(1, total_dialogs // training_args.num_generations)
        samples_per_step = training_args.per_device_train_batch_size * training_args.num_generations
    
    effective_batch_size = samples_per_step * training_args.gradient_accumulation_steps
    training_args.vllm_gpu_memory_utilization = float(grpo_cfg.get("vllm_gpu_memory_utilization", 0.3))
        
    vllm_dev = grpo_cfg.get("vllm_device")
    if isinstance(vllm_dev, str) and vllm_dev:
        training_args.vllm_device = vllm_dev
    
    quantize_8bit = grpo_cfg.get("quantize_8bit", False)
    vllm_dtype = grpo_cfg.get("vllm_dtype")
    
    if quantize_8bit:
        training_args.vllm_dtype = "bfloat16"
    elif isinstance(vllm_dtype, str) and vllm_dtype:
        training_args.vllm_dtype = vllm_dtype
    
    training_args.logging_steps = int(log_cfg.get("logging_steps", 1))
    save_steps = int(log_cfg.get("save_steps", 0))
    training_args.save_steps = save_steps
    training_args.save_strategy = "steps" if save_steps > 0 else "no"
    training_args.save_only_model = True
    training_args.log_completions = bool(log_cfg.get("enable_conversation_log", True))
    training_args.log_on_each_node = bool(log_cfg.get("log_on_each_node", False))
    training_args.report_to = ["wandb"] if wandb_enabled else []

    checkpoint_callback = None
    if TRANSFORMERS_AVAILABLE and int(log_cfg.get("save_steps", 0)) > 0:
        checkpoint_callback = LoRACheckpointCallback()

    return training_args, checkpoint_callback


# Save LoRA adapters, tokenizer, and training config
def save_model_and_config(model, tokenizer, final_checkpoint_dir, cfg, hf_cfg, run_stamp):
    
    try:
        if hasattr(model, "peft_config") and model.peft_config:
            model.save_pretrained(final_checkpoint_dir)
        elif hasattr(model, "save_adapter"):
            model.save_adapter(final_checkpoint_dir)
        else:
            try:
                adapter_state_dict = get_peft_model_state_dict(model)
                adapter_path = os.path.join(final_checkpoint_dir, "adapter_model.safetensors")
                os.makedirs(final_checkpoint_dir, exist_ok=True)
                save_file(adapter_state_dict, adapter_path)
            except Exception as fallback_e:
                model.save_pretrained(final_checkpoint_dir)
        
        tokenizer.save_pretrained(final_checkpoint_dir)
        config_path = os.path.join(final_checkpoint_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2, default=str)
        
        return True
        
    except Exception as e:
        return False


# Upload LoRA adapters to Hugging Face
def push_to_hf_hub(model, hf_cfg, run_stamp, final_checkpoint_dir):
    if not (hf_cfg.get("enabled", False) and hf_cfg.get("push_adapters", False)):
        return  
    base_repo = hf_cfg.get("adapter_repo_base")
    if not base_repo:
        return  
    try:
        adapter_repo_id = f"{base_repo.rstrip('/')}-{run_stamp}"
        create_repo(adapter_repo_id, private=hf_cfg.get("private", True), exist_ok=True)
        
        if hasattr(model, "peft_config") and model.peft_config:
            model.push_to_hub(adapter_repo_id, private=hf_cfg.get("private", True))
        elif hasattr(model, "push_adapter"):
            model.push_adapter(adapter_repo_id, private=hf_cfg.get("private", True))
        else:
            upload_folder(
                folder_path=final_checkpoint_dir,
                repo_id=adapter_repo_id,
                private=hf_cfg.get("private", True),
                ignore_patterns=["*.bin", "model-*.safetensors", "model.safetensors.index.json"]
            )
        
        with open(os.path.join(final_checkpoint_dir, "adapter_repo_id.txt"), "w", encoding="utf-8") as f:
            f.write(adapter_repo_id + "\n")
            
    except Exception as e:
        return
