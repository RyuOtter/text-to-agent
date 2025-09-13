import json
import os
from pathlib import Path
from .utils_accessing import get_user_prompt, get_assistant_prompt
from .utils_sim import Agent, LLMModel
from .run_sim_single import single_simulation
from .improver_utils import summarizer, improver, prompt_change_analysis
from .utils_tools import get_tools
from .sgd_evaluate import evaluate as sgd_evaluate
from .multiwoz_evaluate import evaluate_by_domain as multiwoz_evaluate

# Prompt improvement loop
def improver_loop(config, train_dialogs, schemas):

    # Configuration
    benchmark = config["benchmark"]
    domain = config["domain"]
    prompt_category = config["prompt_category"]
    method = config["improver_method"]["method"]
    batch_size = config["improver_method"]["batch_size"]
    run_dir = config["run_info"]["run_dir"]
    user_model = {"dialogue": LLMModel(**config["llm_models"]["user_model"])}
    assistant_model = {"dialogue": LLMModel(**config["llm_models"]["assistant_model"]),"correction": LLMModel(**config["llm_models"]["assistant_model"])}
    summarizer_model = LLMModel(**config["llm_models"]["summarizer_model"])
    improver_model = LLMModel(**config["llm_models"]["improver_model"])
    analysis_model = LLMModel(**config["llm_models"]["analysis_model"])
    eval_model = LLMModel(**config["llm_models"]["evaluator_model"]) 
    batches = [train_dialogs[i:i+batch_size] for i in range(0, len(train_dialogs), batch_size)]

    # Setup user agent and loop
    benchmark = config["benchmark"]
    user_prompt = get_user_prompt(benchmark, test_mode=True)
    user = Agent(name="user", role="user", system_prompt=user_prompt, model=user_model)

    log_sim_conv = []
    log_tool_logs = []
    log_summary = []
    log_improved_prompt = []
    log_change_analysis = []
    log_metrics = []

    # Loop over batches
    iteration = 0
    
    for batch in batches:

        # Set up assistant agent
        assistant_prompt = get_assistant_prompt(benchmark, domain, prompt_category, iteration="latest", test_mode=True, run_dir=run_dir)
        assistant = Agent(name="assistant", role="assistant", system_prompt=assistant_prompt, model=assistant_model, tools=None)
        if benchmark == "MultiWOZ":
            tools = get_tools(benchmark=benchmark, datapoint=None, schemas=None, dynamic=False)
            assistant.set_tools(tools)
        batch_conversations = []
        batch_tool_logs = []
        batch_eval = []

        # Simulate and evaluate dialogues
        for individual_dialog in batch:
            if benchmark == "SGD":
                simulated_conversation, tool_logs = single_simulation(benchmark, individual_dialog, user, assistant, max_iterations=10, schemas=schemas, ground_truth_included = False)
                try:
                    eval_result, cost = sgd_evaluate(individual_dialog, simulated_conversation, tool_logs, eval_model)
                except Exception as e:
                    print("Evaluation failed")
                    eval_result = {"error": str(e)}
                    cost = 0
            elif benchmark == "MultiWOZ":
                simulated_conversation, tool_logs, goal, goal_messages, dialog_refer = single_simulation(benchmark, individual_dialog, user, assistant, max_iterations=10, schemas=schemas, ground_truth_included = False)
                run_result = {
                    "goals": individual_dialog.get("goal", {}),
                    "dialog_pred": simulated_conversation,
                    "goal_messages": individual_dialog.get("goal", {}).get("message", []),
                    "cost": 0.0
                }
                
                eval_results = {}
                fail_domains = []
                for domain_name in ["restaurant", "hotel", "attraction", "train", "taxi"]:
                    if not individual_dialog["goal"].get(domain_name):
                        continue
                    try:
                        eval_result_dom = multiwoz_evaluate(domain_name, run_result, model=eval_model)
                        eval_results[domain_name] = eval_result_dom
                        run_result["cost"] += eval_result_dom["cost"]
                    except Exception as e:
                        fail_domains.append(domain_name)
                        eval_results[domain_name] = {"exception": f"Run dialog failed as {e.__class__.__name__}: {str(e)}"}
                
                eval_result = eval_results
                cost = run_result["cost"]

            batch_conversations.append(simulated_conversation)
            batch_tool_logs.append(tool_logs)
            
            batch_eval.append(eval_result)

        # Summarizer agent
        summary = summarizer(batch_conversations, batch_tool_logs, summarizer_model)

        # Improver agent
        improved_prompt = improver(method, assistant_prompt, summary, improver_model, summarizer_model)

        # Log metrics
        change_analysis = prompt_change_analysis(improved_prompt, assistant_prompt, analysis_model)
        if benchmark == "SGD":
            total_inform = 0
            total_success = 0
            total_intents = 0
            total_success_intents = 0
            
            for i, eval_res in enumerate(batch_eval):
                if isinstance(eval_res, dict) and "error" not in eval_res:
                    for service_name, service_result in eval_res.items():
                        if isinstance(service_result, dict):
                            for intent_name, intent_result in service_result.items():
                                if isinstance(intent_result, dict):
                                    total_intents += 1
                                    inform_val = intent_result.get("inform", 0)
                                    success_val = intent_result.get("success", 0)
                                    inform_contribution = int(inform_val) if inform_val is not None else 0
                                    total_inform += inform_contribution
                                    
                                    if success_val is not None:
                                        total_success_intents += 1
                                        success_contribution = int(success_val)
                                        total_success += success_contribution
            batch_metrics = {
                "batch_size": len(batch_eval),
                "inform_rate": total_inform / total_intents if total_intents > 0 else 0.0,
                "success_rate": total_success / total_success_intents if total_success_intents > 0 else 0.0,
            }
        elif benchmark == "MultiWOZ":
            all_domains = set()
            for eval_res in batch_eval:
                if "error" not in eval_res:
                    all_domains.update([k for k in eval_res.keys() if k != "total_cost"])
            
            domain_metrics = {}
            for domain in all_domains:
                domain_scores = [eval_res.get(domain, {}) for eval_res in batch_eval if domain in eval_res and "error" not in eval_res]
                if domain_scores:
                    total_score = 0
                    for s in domain_scores:
                        inform = s.get("inform", {}).get("complete", 0) if s.get("inform") else 0
                        success = s.get("success", {}).get("complete", 0) if s.get("success") else None
                        book = s.get("book", {}).get("complete", 0) if s.get("book") else None
                        
                        if success is None and book is None:
                            score = 1.0 * inform
                        elif success is not None and book is None:
                            score = 0.5 * inform + 0.5 * success 
                        elif success is None and book is not None:
                            score = 0.5 * inform + 0.5 * book
                        else:
                            score = 0.5 * inform + 0.25 * success + 0.25 * book
                        
                        total_score += score
                    
                    domain_metrics[domain] = {
                        "count": len(domain_scores),
                        "avg_score": total_score / len(domain_scores) if domain_scores else 0
                    }
            

            batch_metrics = {
                "batch_size": len(batch_eval),
                "domain_metrics": domain_metrics,
            }

        # Save improved prompt
        iteration = iteration + 1
        with open(Path(run_dir) / f"prompt_iteration_{iteration:03d}.txt", "w", encoding="utf-8") as f:
            f.write(improved_prompt)

        log_sim_conv.extend(batch_conversations)
        log_tool_logs.extend(batch_tool_logs)
        log_summary.append(summary)
        log_improved_prompt.append(improved_prompt)
        log_change_analysis.append(change_analysis)
        log_metrics.append(batch_metrics)

    # Save outputs
    with open(Path(run_dir) / "simulated_conversations.jsonl", "w", encoding="utf-8") as f:
        for conv in log_sim_conv:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    
    with open(Path(run_dir) / "tool_logs.jsonl", "w", encoding="utf-8") as f:
        for log in log_tool_logs:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
    
    with open(Path(run_dir) / "conversation_summaries.jsonl", "w", encoding="utf-8") as f:
        for summary in log_summary:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    
    with open(Path(run_dir) / "change_analyses.jsonl", "w", encoding="utf-8") as f:
        for analysis in log_change_analysis:
            f.write(json.dumps(analysis, ensure_ascii=False) + "\n")
    
    with open(Path(run_dir) / "batch_metrics.jsonl", "w", encoding="utf-8") as f:
        for metrics in log_metrics:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")