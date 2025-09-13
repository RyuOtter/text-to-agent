from Modular_Learning_Agent.utils_accessing import multiwoz_data_loader
from Modular_Learning_Agent.multiwoz_data_utils import prepare_goals_string
from Modular_Learning_Agent.utils_tools import get_tools
from Modular_Learning_Agent.run_sim_single import single_simulation
from Modular_Learning_Agent.multiwoz_evaluate import evaluate_by_domain
from Modular_Learning_Agent.utils_sim import Agent
from .MTRA_utils_simulation import single_simulation_rl

# Load training data for MultiWOZ
def multiwoz_train_data_loader(n_datapoints=500, data_seed=42, splits=["train"]):

    split = splits[0] if splits else "train"    
    data_dict = multiwoz_data_loader(split=split, n_points=n_datapoints, random_seed=data_seed)
    
    train_dialogs = []
    for dialog_id, dialog_data in data_dict.items():
        training_dialog = {
            "dialogue_id": dialog_id,
            "dialog_id": dialog_id,  
            "goal": dialog_data.get("goal", {}),
            "log": dialog_data.get("log", []),
            "services": _extract_services_from_goal(dialog_data.get("goal", {})),
            "domain": _get_primary_domain(dialog_data.get("goal", {})),
        }
        train_dialogs.append(training_dialog)
    
    print(f"Loaded {len(train_dialogs)} MultiWOZtraining dialogs")
    return train_dialogs

# Get domains
def _extract_services_from_goal(goal):
    services = []
    for domain in ["restaurant", "hotel", "attraction", "train", "taxi"]:
        if goal.get(domain):
            services.append(domain)
    return services

# Get primary domain
def _get_primary_domain(goal):
    services = _extract_services_from_goal(goal)
    return services[0] if services else "unknown"

# Form dataset into format for env trainer
def build_multiwoz_dataset_and_maps(train_dialogs, assistant_prompt, schemas, benchmark="MultiWOZ"):
    
    prompt_to_dp = {}
    
    for dialog in train_dialogs:
        dialog_id = dialog["dialogue_id"]
        
        goal_messages = dialog.get("goal", {}).get("message", [])
        if goal_messages:
            prompt = prepare_goals_string(goal_messages)
        else:
            prompt = f"Dialog {dialog_id}"
        
        prompt_to_dp[prompt] = dialog
    
    dataset_list = []
    for dialog in train_dialogs:
        goal_messages = dialog.get("goal", {}).get("message", [])
        if goal_messages:
            prompt = prepare_goals_string(goal_messages)
        else:
            prompt = f"Dialog {dialog['dialogue_id']}"
        
        dataset_item = {"prompt": prompt, "raw_datapoint": dialog,}
        dataset_list.append(dataset_item)
        
    return dataset_list, prompt_to_dp


# Adjust simulator for GRPO and MutliWOZ
def create_multiwoz_simulator_function(user_llm, eval_llm, max_turns, user_prompt, assistant_prompt):
    
    # MultiWOZ simulator
    def multiwoz_simulator(**kwargs):

        dp = kwargs.get("datapoint") or kwargs.get("dp")
        assistant_models = kwargs.get("assistant_models", {"dialogue": user_llm})
        user_models = kwargs.get("user_models", {"dialogue": user_llm})
        schemas = kwargs.get("schemas")  
        policy_seed = kwargs.get("policy_seed")
        benchmark = kwargs.get("benchmark", "MultiWOZ")
        
        user_agent = Agent(name="user", role="user", system_prompt=user_prompt, model=user_models)
        assistant_agent = Agent(name="assistant", role="assistant", system_prompt=assistant_prompt, model=assistant_models)
        
        tools = get_tools(benchmark="MultiWOZ", datapoint=None, schemas=None, dynamic=False)
        assistant_agent.set_tools(tools)
        
        try:
            simulated_conversation, function_logs, rl_recorder, assistant_only = single_simulation_rl(
                benchmark="MultiWOZ",
                individual_dialog=dp,
                user=user_agent,
                assistant=assistant_agent,
                max_iterations=max_turns,
                schemas=None,  
                policy_seed=policy_seed,
            )
            
            sim_conv = []
            for turn in simulated_conversation:
                sim_conv.append({
                    "user": turn.get("user", ""),
                    "agent": turn.get("agent", ""), 
                    "turn_idx": turn.get("turn_idx", 0)
                })
                        
            return sim_conv, function_logs, rl_recorder, assistant_only
            
        except Exception as e:
            return [], [], None, ""
    
    return multiwoz_simulator

# Security check
def validate_multiwoz_config(config):
    
    data_cfg = config.get("data", {})
    benchmark = data_cfg.get("benchmark", "SGD")
    if benchmark != "MultiWOZ":
        return True
    multiwoz_cfg = data_cfg.get("multiwoz", {})
    required_fields = ["split", "max_dialogs", "random_seed"]
    return all(field in multiwoz_cfg for field in required_fields)
