from __future__ import annotations
import os
import argparse
import json
import shutil
import sys
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig
import verifiers as vf
from Modular_Learning_Agent.sgd_data_utils import load_schemas
from Modular_Learning_Agent.utils_accessing import sgd_train_data_loader, get_user_prompt, get_assistant_prompt
from Modular_Learning_Agent.utils_llm import LLMModel
from .MTRA_utils_model import load_mtra_model
from .MTRA_utils_config import load_yaml_config, build_dataset_and_maps
from .MTRA_utils_env import SGDEnvAdapter
from .MTRA_utils_env_multiwoz import MultiWOZEnvAdapter
from .MTRA_utils_multiwoz import multiwoz_train_data_loader, build_multiwoz_dataset_and_maps, validate_multiwoz_config, create_multiwoz_simulator_function
from .MTRA_utils_rewards import make_outcome_reward_func, RewardConfig
from .MTRA_utils_rewards_tools import make_turn_reward_func
from .MTRA_utils_rewards_naturalness import make_naturalness_reward_func, make_episodic_naturalness_reward_func
from .MTRA_utils_verify import verify_lora_adapters, enhanced_verify_lora_adapters
from .MTRA_utils_logging import setup_logging_paths, create_logged_reward_function, create_logged_generate_function, save_training_results, setup_wandb, log_training_plots, finish_wandb, get_wandb_enabled
from .MTRA_utils_training import setup_directories, setup_training_args, save_model_and_config, push_to_hf_hub
from .MTRA_utils_simulation import create_simulator_function, create_build_initial_user_text_function, setup_environment_generate, disable_vllm_weight_reloading, enable_vllm_weight_reloading_with_confirmation, setup_vllm_sampling_params

# GRPO training function
def main():
    parser = argparse.ArgumentParser(description="GRPOEnvTrainer (episodic rewards) for SGD and MultiWOZ")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "MTRA_config.yaml"))
    args = parser.parse_args()

    # Loading config
    cfg = load_yaml_config(args.config)
    validate_multiwoz_config(cfg)
    data_cfg = cfg.get("data", {})
    grpo_cfg = cfg.get("grpo", {})
    asst_cfg = cfg.get("assistant_model", {})
    user_cfg = cfg.get("user_model", {})
    eval_cfg = cfg.get("evaluator_model", {})
    log_cfg = cfg.get("logging", {})
    hf_cfg = cfg.get("hf_hub", {})
    benchmark = data_cfg.get("benchmark", "SGD")
    print(f"Benchmark: {benchmark}")
    
    benchmark_naming = log_cfg.get("benchmark_naming", True)
    if benchmark_naming:
        benchmark_prefix = benchmark.lower()
        log_cfg["results_dir_root"] = f"{log_cfg['results_dir_root']}/{benchmark_prefix}"
        log_cfg["checkpoints_root"] = f"{log_cfg['checkpoints_root']}/{benchmark_prefix}"
    
    run_checkpoint_root, run_results_dir, run_stamp = setup_directories(log_cfg)
    
    try:
        results_config_path = os.path.join(run_results_dir, "config.yaml")
        config_file = args.config
        if config_file and os.path.exists(config_file):
            shutil.copy2(config_file, results_config_path)
    except Exception:
        pass

    # Setting up models
    model_name = asst_cfg.get("model_name", "meta-llama/Llama-3.1-8B-Instruct")
    max_seq_len = int(asst_cfg.get("max_seq_length", 6144))
    print(f"Using model: {model_name} with max_seq_length: {max_seq_len}")

    model_kwargs = {}
    quantize_8bit = grpo_cfg.get("quantize_8bit", False)
    
    if asst_cfg.get("load_in_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
        print("4-bit quantization for training model")
    elif asst_cfg.get("load_in_8bit", False) or quantize_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = quantization_config
        print("8-bit quantization for training model")
    
    model, tokenizer = load_mtra_model(model_name, model_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.model_max_length = max_seq_len

    if asst_cfg.get("lora", {}):
        lora_cfg = asst_cfg["lora"]
        peft_config = LoraConfig(
            r=int(lora_cfg.get("r", 4)),
            lora_alpha=int(lora_cfg.get("alpha", 8)),
            lora_dropout=float(lora_cfg.get("dropout", 0.0)),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            task_type="CAUSAL_LM",
        )
        print(f"LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}, dropout={peft_config.lora_dropout}")
    else:
        peft_config = None

    if benchmark == "SGD":
        schemas = load_schemas(data_cfg.get("domain", "sgd"))
        user_prompt = get_user_prompt(benchmark, test_mode=data_cfg.get("test_mode", False))
        assistant_prompt = get_assistant_prompt(
            benchmark,
            data_cfg.get("domain", "sgd"), 
            data_cfg.get("prompt_category", "vanilla"),
            data_cfg.get("iteration", "latest"),
            test_mode=data_cfg.get("test_mode", False)
        )
        train_dialogs = sgd_train_data_loader(
            n=int(data_cfg.get("n_datapoints", 500)),
            seed=int(data_cfg.get("data_seed", 42)),
            splits=data_cfg.get("splits", ["train", "dev"])
        )
        print(f"Loaded {len(train_dialogs)} SGD training dialogs")
        
        train_dataset, prompt_to_dp = build_dataset_and_maps(
            train_dialogs, assistant_prompt, None, schemas, benchmark
        )
        
    elif benchmark == "MultiWOZ":
        schemas = None
        user_prompt = get_user_prompt(benchmark, test_mode=data_cfg.get("test_mode", False))
        assistant_prompt = get_assistant_prompt(
            benchmark,
            "multiwoz",
            data_cfg.get("prompt_category", "vanilla"),
            data_cfg.get("iteration", "latest"),
            test_mode=data_cfg.get("test_mode", False)
        )
        
        train_dialogs = multiwoz_train_data_loader(
            n_datapoints=int(data_cfg.get("n_datapoints", 500)),
            data_seed=int(data_cfg.get("data_seed", 42)),
            splits=data_cfg.get("splits", ["train"])
        )
        print(f"Loaded {len(train_dialogs)} MultiWOZ training dialogs")
        
        train_dataset, prompt_to_dp = build_multiwoz_dataset_and_maps(
            train_dialogs, assistant_prompt, schemas, benchmark
        )
        
    else:
        raise ValueError("Wrong benchmark")

    user_llm = LLMModel(provider=user_cfg.get("provider", "groq"), model_name=user_cfg.get("model_name", "llama-3.1-8b-instant"), temperature=float(user_cfg.get("temperature", 0.0)), max_tokens=int(user_cfg.get("max_tokens", 256)))
    eval_llm = LLMModel(provider=eval_cfg.get("provider", "groq"), model_name=eval_cfg.get("model_name", "llama-3.1-8b-instant"), temperature=float(eval_cfg.get("temperature", 0.0)), max_tokens=int(eval_cfg.get("max_tokens", 256)))

    reward_config_dict = cfg.get("rewards", {}).get("config", {})
    reward_cfg = RewardConfig(w_inform=float(reward_config_dict.get("w_inform", 3.0)), w_success=float(reward_config_dict.get("w_success", 3.0)), w_book=float(reward_config_dict.get("w_book", 3.0)), normalize=False)
    base_outcome_reward = make_outcome_reward_func(reward_cfg, eval_llm, benchmark)
    
    wb_enabled = setup_wandb(cfg, run_stamp, model_name, benchmark)
    
    training_args, checkpoint_callback = setup_training_args(grpo_cfg, log_cfg, model_name, train_dialogs, max_seq_len, wb_enabled, run_results_dir)
    assistant_temperature = float(asst_cfg.get("temperature", 0.8))
    assistant_max_tokens = int(asst_cfg.get("max_tokens", 256))

    # Setting up MTRA env
    if benchmark == "SGD":
        simulator = create_simulator_function(user_llm, eval_llm, int(grpo_cfg.get("max_turns", 8)), user_prompt, assistant_prompt)
    elif benchmark == "MultiWOZ":
        simulator = create_multiwoz_simulator_function(user_llm, eval_llm, int(grpo_cfg.get("max_turns", 8)), user_prompt, assistant_prompt)
    
    build_initial_user_text = create_build_initial_user_text_function()

    if benchmark == "SGD":
        env = SGDEnvAdapter(tokenizer=tokenizer, schemas=schemas, user_models={"dialogue": user_llm}, max_turns=int(grpo_cfg.get("max_turns", 8)))
    elif benchmark == "MultiWOZ":
        env = MultiWOZEnvAdapter(tokenizer=tokenizer, schemas=schemas, user_models={"dialogue": user_llm}, max_turns=int(grpo_cfg.get("max_turns", 8)), benchmark=benchmark)
    else:
        raise ValueError("Wrong benchmark")
    
    # Setup for logging
    hf_cfg = cfg.get("hf_hub", {})
    run_checkpoint_root, run_results_dir, run_stamp = setup_directories(log_cfg)
    metrics_log_path, group_summary_path, convo_log_path, actions_log_path = setup_logging_paths(run_results_dir, log_cfg)
    enable_tools_rewards = cfg.get("rewards", {}).get("enable_tools_rewards", False)
    enable_naturalness_rewards = cfg.get("rewards", {}).get("enable_naturalness_rewards", False)
    outcome_reward_funcs = []
    turn_reward_funcs = []
    
    if enable_tools_rewards:
        turn_config = cfg.get("rewards", {}).get("turn_config", {})         
        tools_reward_func = make_turn_reward_func("rew_cfg_1_tools", turn_config, eval_llm)
        tools_reward_func.__name__ = "tools_outcome_reward"
        outcome_reward_funcs.append(tools_reward_func)
        
    if enable_naturalness_rewards:
        naturalness_config = cfg.get("rewards", {}).get("naturalness_config", {})
        episodic_naturalness_func = make_episodic_naturalness_reward_func("rew_cfg_2_naturalness", naturalness_config, eval_llm, tokenizer)
        outcome_reward_funcs.append(episodic_naturalness_func)
        turn_naturalness_func = make_naturalness_reward_func("rew_cfg_2_naturalness", naturalness_config, eval_llm, tokenizer)
        turn_reward_funcs.append(turn_naturalness_func)
    
    all_outcome_reward_funcs = [base_outcome_reward] + outcome_reward_funcs
    no_turn_reward = len(turn_reward_funcs) == 0
    
    # Creating trainer
    trainer = vf.GRPOEnvTrainer(model=model, processing_class=tokenizer, env=env, turn_reward_funcs=turn_reward_funcs, outcome_reward_funcs=all_outcome_reward_funcs, no_turn_reward=no_turn_reward, args=training_args, train_dataset=train_dataset, peft_config=peft_config, metrics_log_path=metrics_log_path, convo_log_path=convo_log_path, actions_log_path=actions_log_path)
    
    if checkpoint_callback is not None:
        trainer.add_callback(checkpoint_callback)
    
    setup_environment_generate(env, simulator, prompt_to_dp, build_initial_user_text, grpo_cfg, data_cfg, training_args, env.__class__.generate)

    off_policy = grpo_cfg.get("off_policy", True)
    if off_policy:
        disable_vllm_weight_reloading(trainer)
    else:
        enable_vllm_weight_reloading_with_confirmation(trainer)
    
    setup_vllm_sampling_params(trainer, assistant_temperature, assistant_max_tokens, max_seq_len)

    # training step
    trainer.train()

    # Saving and logging
    final_checkpoint_dir = os.path.join(run_results_dir, "final_model")
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    
    try:
        success = save_model_and_config(trainer.model, tokenizer, final_checkpoint_dir, cfg, hf_cfg, run_stamp)
        
        if success:
            save_training_results(run_results_dir, final_checkpoint_dir, trainer, cfg, metrics_log_path, group_summary_path, convo_log_path, actions_log_path, grpo_cfg)

    push_to_hf_hub(trainer.model, hf_cfg, run_stamp, final_checkpoint_dir)

    if os.path.exists(final_checkpoint_dir):
        verify_lora_adapters(final_checkpoint_dir)
        
        results = enhanced_verify_lora_adapters(final_checkpoint_dir, verbose=True)
        if results:
            report_path = os.path.join(final_checkpoint_dir, "lora_verification_report.json")
            with open(report_path, "w") as f:
                json.dump(results, f, indent=2)

    if wb_enabled:
        log_training_plots(trainer, run_results_dir)
        finish_wandb()

# Final execution of main
if __name__ == "__main__":
    main()