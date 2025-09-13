from __future__ import annotations
import gc
import os
import json
import hashlib
import shutil
import sys
from datetime import datetime, timezone
from collections import defaultdict
import torch
import wandb
WANDB_AVAILABLE = True
import matplotlib.pyplot as plt
import numpy as np
MATPLOTLIB_AVAILABLE = True

_GLOBAL_GENERATION_COUNTER = 0
_GLOBAL_ACTION_COUNTER = 0
_LOGGED_CONVERSATIONS = set()
_LOGGED_ACTIONS = set()

# Reset logging counters for new training run
def reset_global_counters():
    global _GLOBAL_GENERATION_COUNTER, _GLOBAL_ACTION_COUNTER, _LOGGED_CONVERSATIONS, _LOGGED_ACTIONS
    _GLOBAL_GENERATION_COUNTER = 0
    _GLOBAL_ACTION_COUNTER = 0
    _LOGGED_CONVERSATIONS.clear()
    _LOGGED_ACTIONS.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Setup file paths for metrics and conversation logging
def setup_logging_paths(run_results_dir, log_cfg):
    reset_global_counters()
    metrics_log_path = os.path.join(run_results_dir, "metrics.jsonl")
    group_summary_path = os.path.join(run_results_dir, "group_summary.jsonl")
    
    with open(metrics_log_path, "w") as f:
        pass
    with open(group_summary_path, "w") as f:
        pass
    
    enable_convo_log = bool(log_cfg.get("enable_conversation_log", True))
    convo_log_path = None
    actions_log_path = None
    if enable_convo_log:
        convo_log_name = log_cfg.get("convo_log_name", "conversations.jsonl")
        convo_log_path = os.path.join(run_results_dir, convo_log_name)
        
        actions_log_name = log_cfg.get("actions_log_name", "actions.jsonl")
        actions_log_path = os.path.join(run_results_dir, actions_log_name)
        
        with open(convo_log_path, "w") as f:
            pass
        with open(actions_log_path, "w") as f:
            pass
    
    return metrics_log_path, group_summary_path, convo_log_path, actions_log_path

# no longer used
def create_logged_reward_function(base_reward_func, metrics_log_path, convo_log_path, turn_reward_funcs=None, outcome_reward_funcs=None):
    return base_reward_func

# no longer used
def create_logged_generate_function(original_generate, actions_log_path):
    return original_generate


# Load data from JSONL
def load_jsonl(file_path):
    data = []
    if not os.path.exists(file_path):
        return data
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


# reward vs step
def plot_reward_vs_step(metrics_data, save_path):
    if not MATPLOTLIB_AVAILABLE:
        return False
        
    steps = []
    rewards = []
    
    for entry in metrics_data:
        if "generation_idx" in entry and "reward" in entry:
            steps.append(entry["generation_idx"])
            rewards.append(entry["reward"])
    
    if not steps:
        return False
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, rewards, "b-o", linewidth=2, markersize=6)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.title("Reward vs Step", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        plt.close()
        return False

# Create loss vs step plot
def plot_loss_vs_step(trainer_data, save_path):
    if not MATPLOTLIB_AVAILABLE:
        return False
        
    steps = []
    losses = []
    
    for entry in trainer_data:
        if "step" in entry and "loss" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])
    
    if not steps:
        return False
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, "r-o", linewidth=2, markersize=6)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Loss vs Step", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        plt.close()
        return False


# Show loss and reward in one plot
def plot_combined_metrics(metrics_data, trainer_data, save_path):
    if not MATPLOTLIB_AVAILABLE:
        return False
        
    reward_steps = []
    rewards = []
    for entry in metrics_data:
        if "generation_idx" in entry and "reward" in entry:
            reward_steps.append(entry["generation_idx"])
            rewards.append(entry["reward"])
    
    loss_steps = []
    losses = []
    for entry in trainer_data:
        if "step" in entry and "loss" in entry:
            loss_steps.append(entry["step"])
            losses.append(entry["loss"])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    if reward_steps:
        ax1.plot(reward_steps, rewards, "b-o", linewidth=2, markersize=6)
        ax1.set_xlabel("Step (Generation Index)", fontsize=12)
        ax1.set_ylabel("Reward", fontsize=12)
        ax1.set_title("Reward vs Step", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No reward data available", ha="center", va="center", transform=ax1.transAxes)
    
    if loss_steps:
        ax2.plot(loss_steps, losses, "r-o", linewidth=2, markersize=6)
        ax2.set_xlabel("Step", fontsize=12)
        ax2.set_ylabel("Loss", fontsize=12)
        ax2.set_title("Loss vs Step", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No loss data available", ha="center", va="center", transform=ax2.transAxes)
    
    plt.tight_layout()
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        plt.close()
        return False


# Save and generate training plots
def generate_training_plots(run_results_dir, trainer):
    if not MATPLOTLIB_AVAILABLE:
        return
    
    metrics_file = os.path.join(run_results_dir, "metrics.jsonl")
    trainer_file = os.path.join(run_results_dir, "trainer_log_history.jsonl")
    
    metrics_data = load_jsonl(metrics_file)
    trainer_data = load_jsonl(trainer_file)
    plots_generated = []
    
    if metrics_data:
        reward_path = os.path.join(run_results_dir, "reward_vs_step.png")
        if plot_reward_vs_step(metrics_data, reward_path):
            plots_generated.append("reward_vs_step.png")
    
    if trainer_data:
        loss_path = os.path.join(run_results_dir, "loss_vs_step.png")
        if plot_loss_vs_step(trainer_data, loss_path):
            plots_generated.append("loss_vs_step.png")
    
    if metrics_data or trainer_data:
        combined_path = os.path.join(run_results_dir, "training_plots.png")
        if plot_combined_metrics(metrics_data, trainer_data, combined_path):
            plots_generated.append("training_plots.png")


# Save training results and create summary files
def save_training_results(run_results_dir, final_checkpoint_dir, trainer, cfg, metrics_log_path, group_summary_path, convo_log_path, actions_log_path, grpo_cfg):
    
    results_config_path = os.path.join(run_results_dir, "config.yaml")
    config_file = None
    for arg in sys.argv:
        if arg.endswith(".yaml") or arg.endswith(".yml"):
            config_file = arg
            break
    if not config_file:
        config_file = "SRC/Finetuning_GRPO/MTRA_config.yaml"
    shutil.copy2(config_file, results_config_path)
    
    if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
        trainer_log_path = os.path.join(run_results_dir, "trainer_log_history.jsonl")
        with open(trainer_log_path, "w") as f:
            for entry in trainer.state.log_history:
                f.write(json.dumps(entry, default=str) + "\n")
        
    with open(group_summary_path, "w") as f:
        pass
        
    if os.path.exists(metrics_log_path):
        dialogue_groups = {}
        
        with open(metrics_log_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        metric_entry = json.loads(line.strip())
                        prompt_key = str(metric_entry.get("generation_idx", 0))
                        if prompt_key not in dialogue_groups:
                            dialogue_groups[prompt_key] = []
                        dialogue_groups[prompt_key].append(metric_entry)
                    except json.JSONDecodeError:
                        continue
            
        prompt_to_generations = {}
        if os.path.exists(convo_log_path):
            with open(convo_log_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            convo_entry = json.loads(line.strip())
                            conversation = convo_entry.get("conversation", [])
                            if conversation:
                                initial_prompt = ""
                                for msg in conversation:
                                    if msg.get("role") == "user":
                                        initial_prompt = msg.get("content", "")[:500]
                                        break
                                if initial_prompt:
                                    prompt_hash = hashlib.md5(initial_prompt.encode()).hexdigest()[:8]
                                    generation_idx = convo_entry.get("generation_idx", 0)
                                    if prompt_hash not in prompt_to_generations:
                                        prompt_to_generations[prompt_hash] = []
                                    prompt_to_generations[prompt_hash].append(generation_idx)
                        except json.JSONDecodeError:
                            continue
            
        dialogue_summaries = {}
        with open(metrics_log_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        metric_entry = json.loads(line.strip())
                        generation_idx = metric_entry.get("generation_idx", 0)
                        dialogue_key = f"dialogue_{generation_idx // 2}"
                        for prompt_hash, gen_indices in prompt_to_generations.items():
                            if generation_idx in gen_indices:
                                dialogue_key = f"dialogue_{prompt_hash}"
                                break
                        if dialogue_key not in dialogue_summaries:
                            dialogue_summaries[dialogue_key] = []
                        dialogue_summaries[dialogue_key].append(metric_entry)
                    except json.JSONDecodeError:
                        continue
            
        dialogue_group_summaries = []
        for dialogue_key, metrics in dialogue_summaries.items():
            if not metrics:
                continue
            rewards = [m.get("reward", 0.0) for m in metrics]
            total_informs = [m.get("total_inform", 0) for m in metrics]
            total_successes = [m.get("total_success", 0) for m in metrics]
            inform_rates = [m.get("inform_rate", 0.0) for m in metrics]
            success_rates = [m.get("success_rate", 0.0) for m in metrics]
            dialogue_summary = {"dialogue_id": dialogue_key, "num_completions": len(metrics), "generation_indices": [m.get("generation_idx", 0) for m in metrics], "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0, "reward_std": (sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards))**0.5 if len(rewards) > 1 else 0.0, "reward_min": min(rewards) if rewards else 0.0, "reward_max": max(rewards) if rewards else 0.0, "inform_mean": sum(total_informs) / len(total_informs) if total_informs else 0.0, "success_mean": sum(total_successes) / len(total_successes) if total_successes else 0.0, "inform_rate_mean": sum(inform_rates) / len(inform_rates) if inform_rates else 0.0, "success_rate_mean": sum(success_rates) / len(success_rates) if success_rates else 0.0, "best_completion_idx": rewards.index(max(rewards)) if rewards else 0, "best_reward": max(rewards) if rewards else 0.0, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
            dialogue_group_summaries.append(dialogue_summary)
        
        dialogue_group_summaries.sort(key=lambda x: x["dialogue_id"])
        with open(group_summary_path, "w") as f:
            for summary in dialogue_group_summaries:
                f.write(json.dumps(summary) + "\n")
        
    generate_training_plots(run_results_dir, trainer)
        
    plot_files = ["reward_vs_step.png", "loss_vs_step.png", "training_plots.png"]
    for plot_file in plot_files:
        plot_path = os.path.join(run_results_dir, plot_file)
    
    
# Not used
def setup_wandb(cfg, run_stamp, model_name, benchmark="SGD"):
    if not WANDB_AVAILABLE:
        return False
        
    wb_cfg = cfg.get("wandb", {})
    if not wb_cfg.get("enabled", False):
        return False
    
    try:
        os.environ["WANDB_MODE"] = wb_cfg.get("mode", "online")
        base_model_name = model_name.split("/")[-1] if "/" in model_name else model_name
        run_name = f"mtra-{base_model_name}-{run_stamp}"
        wandb.init(project=wb_cfg.get("project", "mtra-grpo"), entity=wb_cfg.get("entity"), name=run_name, tags=wb_cfg.get("tags", ["MTRA", "GRPO", "RL"]) + [benchmark], config={"model": {"name": model_name, "base_model": model_name}, "lora": cfg.get("lora", {}), "reward": cfg.get("reward", {}), "grpo": cfg.get("grpo", {}), "data": cfg.get("data", {}), "full_config": cfg, "run_stamp": run_stamp}, dir=wb_cfg.get("dir", "./wandb_logs"))
        return True
        
    except Exception as e:
        return False

# Not used
def log_training_plots(trainer, run_results_dir):
    if not WANDB_AVAILABLE or not wandb.run:
        return
        
    try:
        if hasattr(trainer, "plot") and callable(trainer.plot):
            fig = trainer.plot()
            if fig:
                wandb.log({"training_plots": wandb.Image(fig)})
        
        plot_files = ["training_progress.png", "reward_curves.png", "loss_curves.png", "reward_vs_step.png", "loss_vs_step.png", "training_plots.png"]
        
        for plot_file in plot_files:
            plot_path = os.path.join(run_results_dir, plot_file)
            if os.path.exists(plot_path):
                wandb.log({plot_file.replace(".png", ""): wandb.Image(plot_path)})
                
    except Exception as e:
        print(f"Logging to wandb failed: {e}")

# Not used
def finish_wandb():
    if WANDB_AVAILABLE and wandb.run:
        wandb.finish()

# Check if wandb activate
def get_wandb_enabled(cfg):
    return WANDB_AVAILABLE and cfg.get("wandb", {}).get("enabled", False)