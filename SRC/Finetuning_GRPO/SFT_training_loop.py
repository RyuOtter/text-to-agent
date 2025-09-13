import datetime
import json
import os
import sys
import argparse
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments, TrainerCallback, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer
from datasets import Dataset as HFDataset
sys.path.append("SRC")
sys.path.append("SRC/Multi-Turn-RL-Agent")
from Finetuning_GRPO.MTRA_utils_config import load_yaml_config
from Finetuning_GRPO.MTRA_utils_training import push_to_hf_hub

# Configuration
DEFAULT_CONFIG_PATH = "SRC/Finetuning_GRPO/MTRA_config.yaml"
DEFAULT_DATASET_PATH = "SGD_SFT_Data.jsonl"

# Used to feed input and target into SFTTrainer. Tokenizes and concatenates them while masking non-relevant context
class SGDSFTCollator:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        input_ids_list, labels_list, attn_list = [], [], []
        
        for sample in batch:
            ctx_tokens = self.tokenizer(
                sample["input_text"],
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_len // 2,
            )
            tgt_tokens = self.tokenizer(
                sample["target_text"],
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_len - len(ctx_tokens["input_ids"]),
            )
            ctx_ids = torch.tensor(ctx_tokens["input_ids"], dtype=torch.long)
            tgt_ids = torch.tensor(tgt_tokens["input_ids"], dtype=torch.long)
            full_ids = torch.cat([ctx_ids, tgt_ids], dim=0)
            labels = torch.cat([torch.full_like(ctx_ids, -100), tgt_ids.clone()], dim=0)
            attn_mask = torch.ones_like(full_ids)
            full_ids = full_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            attn_mask = attn_mask[:self.max_seq_len]

            input_ids_list.append(full_ids)
            labels_list.append(labels)
            attn_list.append(attn_mask)

        max_len = max(x.size(0) for x in input_ids_list)
        
        def pad_to_max(tensor_list, pad_value):
            padded = []
            for tensor in tensor_list:
                if tensor.size(0) < max_len:
                    padding = torch.full((max_len - tensor.size(0),), pad_value, dtype=tensor.dtype)
                    tensor = torch.cat([tensor, padding], dim=0)
                padded.append(tensor)
            return torch.stack(padded, dim=0)

        return {
            "input_ids": pad_to_max(input_ids_list, self.tokenizer.pad_token_id),
            "labels": pad_to_max(labels_list, -100),
            "attention_mask": pad_to_max(attn_list, 0),
        }


# SFT training argument setup with exactly the same parameters as GRPO
def setup_sft_training_args(grpo_cfg, log_cfg, model_name, dataset_size, max_seq_len, wandb_enabled=False, run_results_dir=None):
    
    grpo_batch = int(grpo_cfg.get("per_device_train_batch_size", 2))
    grpo_sims = 2
    grpo_turns_avg = 8
    grpo_tokens_per_update = grpo_batch * grpo_sims * grpo_turns_avg
    target_batch_size = 32
    gradient_accumulation_steps = int(grpo_cfg.get("gradient_accumulation_steps", 1))
    per_device_batch_size = 8
    gradient_accumulation_steps = target_batch_size // per_device_batch_size
    num_epochs = int(grpo_cfg.get("num_epochs", 1))
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps
    max_steps = num_epochs * (dataset_size // effective_batch_size)

    output_dir = run_results_dir if run_results_dir else "sft_checkpoints"
    if run_results_dir:
        checkpoints_dir = os.path.join(run_results_dir, "checkpoints")
        output_dir = checkpoints_dir
        os.makedirs(checkpoints_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=float(grpo_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(grpo_cfg.get("optimizer", {}).get("weight_decay", 0.0)),
        optim=grpo_cfg.get("optimizer", {}).get("name", "adamw_torch"),
        adam_beta1=float(grpo_cfg.get("optimizer", {}).get("betas", [0.9, 0.999])[0]),
        adam_beta2=float(grpo_cfg.get("optimizer", {}).get("betas", [0.9, 0.999])[1]),
        adam_epsilon=float(grpo_cfg.get("optimizer", {}).get("eps", 1e-8)),
        lr_scheduler_type=grpo_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=0.03,
        bf16=bool(grpo_cfg.get("bf16", True)) and torch.cuda.is_available(),
        fp16=bool(grpo_cfg.get("fp16", False)) and torch.cuda.is_available() and not bool(grpo_cfg.get("bf16", True)),
        tf32=True if torch.cuda.is_available() else False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        max_grad_norm=float(grpo_cfg.get("max_grad_norm", 2.0)),
        ddp_find_unused_parameters=False,
        save_safetensors=True,
        logging_steps=int(log_cfg.get("logging_steps", 1)),
        save_steps=int(log_cfg.get("save_steps", 50)),
        save_strategy="steps" if int(log_cfg.get("save_steps", 50)) > 0 else "no",
        save_total_limit=2,
        report_to=[],
        run_name=f"sft-sgd-{model_name.split('/')[-1].lower()}",
        remove_unused_columns=False,
    )
    
    return training_args


# Main SFT training function using GRPO configuration and SGD dataset
def main():
    parser = argparse.ArgumentParser(description="SFT Training for SGD using exact GRPO parameters")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    grpo_cfg = cfg.get("grpo", {})
    asst_cfg = cfg.get("assistant_model", {})
    log_cfg = cfg.get("logging", {})
    hf_cfg = cfg.get("hf_hub", {})
    model_name = asst_cfg.get("model_name", "meta-llama/Llama-3.1-8B-Instruct")
    max_seq_len = int(asst_cfg.get("max_seq_length", 4200))
    run_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_root = log_cfg.get("results_dir_root", "./Results")
    run_results_dir = os.path.join(results_root, "sft", run_stamp)
    os.makedirs(run_results_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    if model.dtype != model_dtype:
        model = model.to(dtype=model_dtype)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = max_seq_len

    lora_cfg = asst_cfg["lora"]
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        task_type="CAUSAL_LM",
    )

    data_list = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line))
    sft_dataset = HFDataset.from_list(data_list)
    sample_types = {}
    for sample in data_list:
        sample_types[sample["sample_type"]] = sample_types.get(sample["sample_type"], 0) + 1

    wb_enabled = False
    training_args = setup_sft_training_args(
        grpo_cfg, log_cfg, model_name, len(sft_dataset), max_seq_len, wb_enabled, run_results_dir
    )
    data_collator = SGDSFTCollator(tokenizer, max_seq_len)

    # Formatting helper function for SFTTrainer
    def formatting_func(example):
        return example["input_text"] + example["target_text"]

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        peft_config=peft_config,
        train_dataset=sft_dataset,
        data_collator=data_collator,
        formatting_func=formatting_func,
    )
    jsonl_log_path = os.path.join(run_results_dir, "training_metrics.jsonl")
    
    # Logging of training metrics to JSONL
    class JSONLLoggingCallback(TrainerCallback):
        def __init__(self, jsonl_path):
            self.jsonl_path = jsonl_path
            self.jsonl_file = None
            
        def on_train_begin(self, args, state, control, **kwargs):
            self.jsonl_file = open(self.jsonl_path, "w")
            
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and self.jsonl_file:
                log_entry = {"step": state.global_step, **logs}
                self.jsonl_file.write(json.dumps(log_entry) + "\n")
                self.jsonl_file.flush()
                
        def on_train_end(self, args, state, control, **kwargs):
            if self.jsonl_file:
                self.jsonl_file.close()
    
    jsonl_callback = JSONLLoggingCallback(jsonl_log_path)
    trainer.add_callback(jsonl_callback)
    
    # Training steps
    trainer.train()

    try:
        training_logs = []
        with open(jsonl_log_path, "r") as f:
            for line in f:
                training_logs.append(json.loads(line.strip()))
        
        if training_logs and "loss" in training_logs[0]:
            losses = [log["loss"] for log in training_logs]
            steps = [log["step"] for log in training_logs]
            min_loss, max_loss = min(losses), max(losses)
            if "mean_token_accuracy" in training_logs[0]:
                initial_acc = training_logs[0]["mean_token_accuracy"] * 100
                final_acc = training_logs[-1]["mean_token_accuracy"] * 100
                
    except Exception:
        print("Training completed successfully!")
    
    final_checkpoint_dir = os.path.join(run_results_dir, "final_model")
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    try:
        if hasattr(trainer.model, "peft_config") and trainer.model.peft_config:
            trainer.model.save_pretrained(final_checkpoint_dir)
        else:
            trainer.model.save_pretrained(final_checkpoint_dir)
        tokenizer.save_pretrained(final_checkpoint_dir)
        config_path = os.path.join(final_checkpoint_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2, default=str)
        
    except Exception as e:
        print(f"Model saving failed: {e}")

    sft_hf_cfg = hf_cfg.copy()
    if "grpo" in sft_hf_cfg.get("adapter_repo_base", ""):
        sft_hf_cfg["adapter_repo_base"] = sft_hf_cfg["adapter_repo_base"].replace("grpo", "sft")
    push_to_hf_hub(trainer.model, sft_hf_cfg, run_stamp, final_checkpoint_dir)

if __name__ == "__main__":
    main()
