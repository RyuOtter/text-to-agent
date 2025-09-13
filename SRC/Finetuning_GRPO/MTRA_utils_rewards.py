from __future__ import annotations
import json, math, re
from Modular_Learning_Agent.sgd_evaluate import evaluate
from Modular_Learning_Agent.multiwoz_evaluate import evaluate_by_domain

# Overwritten later
class RewardConfig:
    def __init__(self, w_inform=3.0, w_success=3.0, w_book=3.0, normalize=True):
        self.w_inform = w_inform
        self.w_success = w_success
        self.w_book = w_book
        self.normalize = normalize

# Function to map inform/success output to 0 and 1
def _safe01(x):
    if x is None:
        return 0
    if isinstance(x, dict):
        return 0
    try:
        v = float(x)
        if math.isnan(v):
            return 0
        return 1 if v >= 1.0 else 0
    except Exception:
        return 1 if str(x).strip() in {"1", "true", "True"} else 0

# Function to map inform/success output to 0 and 1 (null not considered)
def _safe01_with_incomplete(x):
    if x is None:
        return 0, False
    if isinstance(x, dict):
        return 0, False
    try:
        v = float(x)
        if math.isnan(v):
            return 0, False
        return (1 if v >= 1.0 else 0), False
    except Exception:
        return (1 if str(x).strip() in {"1", "true", "True"} else 0), False

# Reward for SGD eval of task completion
def sgd_reward(eval_result, config=RewardConfig(), *, return_breakdown=False):
    
    total = 0.0
    max_total = 0.0
    breakdown = {}

    num_intents = 0
    all_informs_one = True
    all_success_one = True
    incomplete_count = 0
    
    for service, intents in (eval_result or {}).items():
        for intent, scores in (intents or {}).items():
            num_intents += 1
            
            inform_value = scores.get("inform", 0)
            success_value = scores.get("success")
            
            inform = 1 if inform_value == 1 else 0
            
            if success_value is None and inform == 1:
                success = 1
                intent_score = config.w_inform * inform + config.w_success * success
            elif success_value is None and inform == 0:
                success = 0
                intent_score = config.w_inform * inform + config.w_success * success
            else:
                success = 1 if success_value == 1 else 0
                intent_score = config.w_inform * inform + config.w_success * success
            
            intent_max = config.w_inform + config.w_success

            total += intent_score
            max_total += intent_max

            breakdown.setdefault(service, {})[intent] = {
                "inform": inform,
                "success": success,
                "score": intent_score,
                "max": intent_max,
            }

            all_informs_one = all_informs_one and (inform == 1)
            all_success_one = all_success_one and (success == 1 and success_value == 1)

    if config.normalize and max_total > 0:
        reward = total / max_total
    else:
        reward = total

    return (reward, breakdown) if return_breakdown else (reward, None)

# Turn tool loo a list
def _extract_tool_logs_from_sim_conv(sim_conv):
    logs = []
    
    for turn in sim_conv:
        user_content = turn.get("user", "")
        if user_content and isinstance(user_content, str):
            result_pattern = r"<result>(.*?)</result>"
            matches = re.findall(result_pattern, user_content, re.DOTALL)
            
            for match in matches:
                try:
                    json_content = match.strip()
                    if json_content:
                        parsed_log = json.loads(json_content)
                        logs.append(parsed_log)
                    else:
                        logs.append({})
                        
                except json.JSONDecodeError as e:
                    logs.append({})
                    
                except Exception as e:
                    logs.append({})
    
    return logs

# Reward computation for multiwoz
def multiwoz_reward(eval_result, config=RewardConfig(), *, return_breakdown=False):
    total = 0.0
    max_total = 0.0
    breakdown = {}
    
    num_domains = 0
    all_informs_one = True
    all_success_one = True
    
    for domain, domain_result in (eval_result or {}).items():
        if isinstance(domain_result, dict) and "inform" in domain_result:
            num_domains += 1
            
            inform = _safe01(domain_result.get("inform", {}).get("complete"))
            success = _safe01(domain_result.get("success", {}).get("complete"))
            book = _safe01(domain_result.get("book", {}).get("complete"))
            
            domain_score = config.w_inform * inform + config.w_success * success + config.w_book * book
            domain_max = config.w_inform + config.w_success + config.w_book
            
            total += domain_score
            max_total += domain_max
            
            breakdown[domain] = {
                "inform": inform,
                "success": success,
                "book": book,
                "score": domain_score,
                "max": domain_max,
            }
            
            all_informs_one = all_informs_one and (inform == 1)
            all_success_one = all_success_one and (success == 1)
    
    all_books_one = True
    for domain, domain_result in (eval_result or {}).items():
        if isinstance(domain_result, dict) and "inform" in domain_result:
            book = _safe01(domain_result.get("book", {}).get("complete"))
            all_books_one = all_books_one and (book == 1)
    
    # if all_informs_one and all_success_one and all_books_one and (total > 0):
    #     total += 3.0
    
    if config.normalize and max_total > 0:
        reward = total / max_total
    else:
        reward = total
    
    return (reward, breakdown) if return_breakdown else (reward, None)

# Final reward functions
def make_outcome_reward_func(reward_cfg, eval_llm, benchmark="SGD"):
    
    if benchmark == "SGD":
        return _make_sgd_outcome_reward_func(reward_cfg, eval_llm)
    elif benchmark == "MultiWOZ":
        return _make_multiwoz_outcome_reward_func(reward_cfg, eval_llm)
    else:
        raise ValueError("Wrong benchmark")

# SGD reward function for task com
def _make_sgd_outcome_reward_func(reward_cfg, eval_llm):

    def outcome_reward(*, prompts, completions, raw_datapoint=None, **kwargs):
        rewards = []
        
        detailed_metrics = []
        
        for i in range(len(completions)):
            dp = raw_datapoint[i] if raw_datapoint else {}
            sim_conv = completions[i]
            
            tool_logs_list = kwargs.get("tool_logs", [])
            turn_tool_logs = tool_logs_list[i] if i < len(tool_logs_list) else []
            
            try:
                eval_result, _ = evaluate(dp, sim_conv, turn_tool_logs, eval_llm)
                r_raw, breakdown = sgd_reward(eval_result, reward_cfg, return_breakdown=True)
                rewards.append(float(r_raw))
            except Exception as e:
                r_raw = 3.0
                breakdown = {}
                rewards.append(float(r_raw))
            
            total_inform = 0
            total_success = 0
            total_intents = 0
            total_success_applicable = 0
            
            individual_informs = []
            individual_successes = []
            intent_details = []
            
            if breakdown:
                for service, intents in breakdown.items():
                    for intent, scores in intents.items():
                        inform_score = scores.get("inform", 0)
                        success_score = scores.get("success", 0)
                        
                        original_success = None
                        if eval_result and service in eval_result and intent in eval_result[service]:
                            original_success = eval_result[service][intent].get("success")
                        
                        total_inform += inform_score
                        total_intents += 1
                        
                        if original_success is not None:
                            total_success += success_score
                            total_success_applicable += 1
                            individual_successes.append(success_score)
                        else:
                            individual_successes.append(None)
                        
                        individual_informs.append(inform_score)
                        
                        intent_details.append({
                            "service": service,
                            "intent": intent,
                            "inform": inform_score,
                            "success": success_score if original_success is not None else None,
                            "success_applicable": original_success is not None
                        })
            
            detailed_metrics.append({
                "total_inform": total_inform,
                "total_success": total_success,
                "total_intents": total_intents,
                "total_success_applicable": total_success_applicable,
                "inform_rate": total_inform / max(total_intents, 1),
                "success_rate": total_success / max(total_success_applicable, 1),
                "individual_informs": individual_informs,
                "individual_successes": individual_successes,
                "intent_details": intent_details,
                "breakdown": breakdown
            })
        
        outcome_reward._last_detailed_metrics = detailed_metrics
        
        return rewards

    outcome_reward.__name__ = "sgd_outcome_reward"
    return outcome_reward

# MultiWOZ reward function for task completion
def _make_multiwoz_outcome_reward_func(reward_cfg, eval_llm):

    def outcome_reward(*, prompts, completions, raw_datapoint=None, **kwargs):
        rewards = []
        
        detailed_metrics = []
        
        for i in range(len(completions)):
            dp = raw_datapoint[i] if raw_datapoint else {}
            sim_conv = completions[i]
            
            run_result = {
                "goals": dp.get("goal", {}),
                "dialog_pred": sim_conv,
                "goal_messages": dp.get("goal", {}).get("message", []),
                "cost": 0.0
            }
            
            eval_results = {}
            try:
                for domain_name in ["restaurant", "hotel", "attraction", "train", "taxi"]:
                    domain_goal = dp.get("goal", {}).get(domain_name)
                    if not domain_goal:
                        continue
                    
                    if isinstance(domain_goal, dict):
                        has_content = any(
                            v for v in domain_goal.values() 
                            if v and (not isinstance(v, (dict, list)) or (isinstance(v, (dict, list)) and v))
                        )
                        if not has_content:
                            continue
                    
                    try:
                        eval_result_dom = evaluate_by_domain(domain_name, run_result, model=eval_llm, verbose=False)
                        eval_results[domain_name] = eval_result_dom
                    except Exception as e:
                        print(f"MultiWOZ eval failed:{e}")
                        continue
                        
            except Exception as e:
                rewards.append(0.0)
                breakdown = {}
            else:
                r_raw, breakdown = multiwoz_reward(eval_results, reward_cfg, return_breakdown=True)
                rewards.append(float(r_raw))
            
            total_inform = 0
            total_success = 0
            total_book = 0
            total_domains = 0
            
            domain_details = []
            if breakdown:
                for domain, scores in breakdown.items():
                    inform_score = scores.get("inform", 0)
                    success_score = scores.get("success", 0)
                    book_score = scores.get("book", 0)
                    
                    total_inform += inform_score
                    total_success += success_score
                    total_book += book_score
                    total_domains += 1
                    
                    domain_details.append({"domain": domain,"inform": inform_score,"success": success_score,"book": book_score})
            
            multiwoz_metrics = {
                "total_inform": total_inform,
                "total_success": total_success,
                "total_intents": total_domains,
                "total_success_applicable": total_domains,
                "inform_rate": total_inform / max(total_domains, 1),
                "success_rate": total_success / max(total_domains, 1),
                "individual_informs": [d["inform"] for d in domain_details],
                "individual_successes": [d["success"] for d in domain_details],
                "intent_details": [{"service": d["domain"], "intent": "MultiWOZ", "inform": d["inform"], "success": d["success"], "success_applicable": True} for d in domain_details],
                
                "total_book": total_book,
                "total_domains": total_domains,
                "book_rate": total_book / max(total_domains, 1),
                "individual_books": [d["book"] for d in domain_details],
                "domain_details": domain_details,
                "breakdown": breakdown
            }
            
            detailed_metrics.append(multiwoz_metrics)
        
        outcome_reward._last_detailed_metrics = detailed_metrics
        
        return rewards

    outcome_reward.__name__ = "multiwoz_outcome_reward"
    return outcome_reward