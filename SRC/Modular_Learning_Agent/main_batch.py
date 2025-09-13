import json
import os
from datetime import datetime
from pathlib import Path
from .run_sim_single import single_simulation
from .utils_sim import Agent, LLMModel
from .utils_accessing import sgd_eval_data_loader, get_user_prompt, get_assistant_prompt, multiwoz_data_loader
from .utils_tools import get_tools
from .sgd_data_utils import load_schemas
from .sgd_evaluate import evaluate
from .multiwoz_evaluate import evaluate_by_domain
from .multiwoz_data_utils import json_default_func
import subprocess
import sys

# Configuration
benchmark = "MultiWOZ"
domain = "hotels"
prompt_category = "vanilla"

# Models
eval_llm = LLMModel(provider="openai", model_name="gpt-3.5-turbo", temperature=0.0)

# Option 1: HuggingFace with LoRA for trained models
"""
assistant_dialogue_llm = LLMModel(
    provider="huggingface", 
    model_name="meta-llama/Llama-3.1-8B-Instruct", 
    temperature=0.4, 
    max_tokens=8000,
    lora_adapter_path="fill in model path"
)
assistant_correction_llm = LLMModel(
    provider="huggingface", 
    model_name="meta-llama/Llama-3.1-8B-Instruct", 
    temperature=0.0, 
    max_tokens=8000,
    lora_adapter_path="fill in model path"
)
"""
# Option 2: Groq for Llama-3.1-8b-instant off the shelf
assistant_dialogue_llm = LLMModel(provider="groq", model_name="llama-3.1-8b-instant", temperature=0.4, max_tokens=8000)
assistant_correction_llm = LLMModel(provider="groq", model_name="llama-3.1-8b-instant", temperature=0.0, max_tokens=8000)

assistant_model = {"dialogue": assistant_dialogue_llm, "correction": assistant_correction_llm}

user_dialogue_llm = LLMModel(provider="groq", model_name="llama-3.1-8b-instant", temperature=0.0)
user_model = {"dialogue": user_dialogue_llm}

seed = 789
n_datapoints = 100
split = "test"

# Evaluation on a batch of datapoints
def main(domain, prompt_category, n_datapoints, seed):

    module_root = Path(__file__).resolve().parents[2]
    default_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_run_dir_env = os.environ.get("EVAL_RUN_DIR")
    if eval_run_dir_env:
        eval_run_dir = Path(eval_run_dir_env)
    else:
        benchmark_prefix = "sgd" if benchmark == "SGD" else "multiwoz"
        eval_run_dir = module_root / "Evaluation" / f"{benchmark_prefix}_eval_{default_run_id}"
    eval_run_dir.mkdir(parents=True, exist_ok=True)

    # Load evaluation data
    if benchmark == "SGD":
        eval_dialogs = sgd_eval_data_loader(n=n_datapoints, seed=seed)
        schemas = load_schemas()
    elif benchmark == "MultiWOZ":
        data_dict = multiwoz_data_loader(split=split, n_points=n_datapoints, random_seed=seed)
        eval_dialogs = [{"dialog_id": dialog_id, **dialog_data} for dialog_id, dialog_data in data_dict.items()]
        schemas = None
    else:
        raise ValueError("Wrong benchmark")

    # Initialize agents
    user_prompt = get_user_prompt(benchmark, test_mode=True)
    init_user = Agent(name="user", role="user", system_prompt=user_prompt, model=user_model)
    
    assistant_prompt = get_assistant_prompt(
        benchmark=benchmark,
        domain=domain,
        prompt_category=prompt_category,
        iteration="latest",
        test_mode=True
    )
    init_assistant = Agent(name="assistant", role="assistant", system_prompt=assistant_prompt, model=assistant_model, tools=None)
    
    if benchmark == "MultiWOZ":
        tools = get_tools(benchmark=benchmark, datapoint=None, schemas=None, dynamic=False)
        init_assistant.set_tools(tools)

    # Logging
    domain_counts = {}
    output_jsonl = str(eval_run_dir / "evaluations.jsonl")
    with open(output_jsonl, "w") as _:
        pass

    run_config = {
        "assistant_dialogue": {
            "model_name": assistant_model["dialogue"].model_name,
            "temperature": assistant_model["dialogue"].temperature,
            "max_tokens": getattr(assistant_model["dialogue"], "max_tokens", None),
        },
        "assistant_correction": {
            "model_name": assistant_model["correction"].model_name,
            "temperature": assistant_model["correction"].temperature,
            "max_tokens": getattr(assistant_model["correction"], "max_tokens", None),
        },
        "user_dialogue": {
            "model_name": user_model["dialogue"].model_name,
            "temperature": user_model["dialogue"].temperature,
            "max_tokens": getattr(user_model["dialogue"], "max_tokens", None),
        },
        "eval": {
            "model_name": eval_llm.model_name,
            "temperature": eval_llm.temperature,
            "max_tokens": getattr(eval_llm, "max_tokens", None),
        },
        "n_datapoints": n_datapoints,
        "seed": seed,
    }
    with open(eval_run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # Evaluation loop
    for i, data in enumerate(eval_dialogs):
        if benchmark == "SGD":
            dialog_id = data["dialogue_id"]
        elif benchmark == "MultiWOZ":
            dialog_id = data["dialog_id"]
        
        print(f"Running dialogue {i+1}/{len(eval_dialogs)}: {dialog_id}")
        
        if benchmark == "SGD":
            dialog_domain = data.get("domain") or (data.get("services") or [None])[0]
        elif benchmark == "MultiWOZ":
            active_domains = [domain for domain in ["restaurant", "hotel", "attraction", "train", "taxi"] 
                            if data.get("goal", {}).get(domain)]
            dialog_domain = active_domains[0] if active_domains else "unknown"
        
        if dialog_domain is not None:
            domain_counts[dialog_domain] = domain_counts.get(dialog_domain, 0) + 1

        try:
            if benchmark == "SGD":
                simulated_conversation, tool_logs = single_simulation(
                    benchmark=benchmark,
                    individual_dialog=data,
                    user=init_user,
                    assistant=init_assistant,
                    schemas=schemas,
                    max_iterations=10
                )
            elif benchmark == "MultiWOZ":
                simulated_conversation, tool_logs, goal, goal_messages, dialog_refer = single_simulation(
                    benchmark=benchmark,
                    individual_dialog=data,
                    user=init_user,
                    assistant=init_assistant,
                    schemas=schemas,
                    max_iterations=10
                )
        except Exception as e:
            print("Simulation failed")
            simulated_conversation, tool_logs = [], []
            status = "failed at simulation"
            eval_result, cost = None, 0.0
        else:
            try:
                if benchmark == "SGD":
                    eval_result, cost = evaluate(data, simulated_conversation, tool_logs, eval_llm)
                    status = "succeed"
                    if eval_result:
                        inform_score = eval_result.get("inform", "N/A")
                        success_score = eval_result.get("success", "N/A")
                        print(f"Evaluation: inform={inform_score}, success={success_score}")
                    else:
                        print("Evaluation completed")
                        
                elif benchmark == "MultiWOZ":
                    run_result = {
                        "goals": data.get("goal", {}),
                        "dialog_pred": simulated_conversation,
                        "goal_messages": data.get("goal", {}).get("message", []),
                        "cost": 0.0
                    }
                    
                    eval_results = {}
                    fail_domains = []
                    for domain_name in ["restaurant", "hotel", "attraction", "train", "taxi"]:
                        if not data["goal"].get(domain_name):
                            continue
                        try:
                            eval_result_dom = evaluate_by_domain(domain_name, run_result, model=eval_llm)
                            eval_results[domain_name] = eval_result_dom
                            run_result["cost"] += eval_result_dom["cost"]
                        except Exception as e:
                            fail_domains.append(domain_name)
                            eval_results[domain_name] = {"exception": f"Run dialog failed as {e.__class__.__name__}: {str(e)}"}
                    
                    status = "failed on eval dialog of domain: " + ", ".join(fail_domains) if fail_domains else "succeed"
                    eval_result = eval_results
                    cost = run_result["cost"]
                    
                    if eval_results:
                        domain_scores = []
                        for domain_name, domain_result in eval_results.items():
                            if isinstance(domain_result, dict) and "inform" in domain_result:
                                inform = domain_result["inform"].get("complete", "N/A")
                                success = domain_result["success"].get("complete", "N/A")
                                domain_scores.append(f"{domain_name}(I:{inform},S:{success})")
                    else:
                        print("Evaluation completed")
                        
            except Exception as e:
                print("Evaluation failed")
                eval_result, cost = None, 0.0
                status = "failed at evaluation"

        eval_output = {
            "dialog_id": dialog_id,
            "status": status,
            "eval_results": eval_result,
            "cost": cost,
            "run_result": simulated_conversation
        }

        with open(output_jsonl, "a") as f:
            if benchmark == "MultiWOZ":
                f.write(json.dumps(eval_output, default=json_default_func) + "\n")
            else:
                f.write(json.dumps(eval_output) + "\n")
    
    # Result analysis
    if benchmark == "SGD":
        analysis_script = "sgd_result_analysis"
        score_file = f"sgd_eval_scores.{default_run_id}.md"
    elif benchmark == "MultiWOZ":
        analysis_script = "multiwoz_result_analysis"
        score_file = f"multiwoz_eval_scores.{default_run_id}.md"
    
    env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    
    analysis_cmd = [
        sys.executable, "-m", f"Modular_Learning_Agent.{analysis_script}",
        "--log_file", str(eval_run_dir / "evaluations.jsonl"),
        "--score_table_file", str(eval_run_dir / score_file)
    ]
    
    subprocess.run(analysis_cmd, env=env)

if __name__ == "__main__":
    main(domain, prompt_category, n_datapoints, seed)