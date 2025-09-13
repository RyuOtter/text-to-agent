import json
import sys
from pathlib import Path
sys.path.append("SRC")
from Modular_Learning_Agent.sgd_data_utils import load_schemas
from Modular_Learning_Agent.utils_accessing import sgd_train_data_loader, get_assistant_prompt
from Modular_Learning_Agent.utils_tools import get_tools
from Modular_Learning_Agent.utils_sim import serialize_tools_for_prompt
from Finetuning_GRPO.MTRA_utils_config import load_yaml_config

# Configuration
CONFIG_PATH = "SRC/Finetuning_GRPO/MTRA_config.yaml"
OUTPUT_JSONL = "SGD_SFT_Data.jsonl"


# Load exact same SGD dialogues as GRPO training
def load_exact_same_sgd_data(config):
    data_cfg = config["data"]
    n_datapoints = int(data_cfg.get("n_datapoints", 500))
    data_seed = int(data_cfg.get("data_seed", 42))
    splits = data_cfg.get("splits", ["train", "dev"])
    train_dialogs = sgd_train_data_loader(
        n=n_datapoints,
        seed=data_seed,
        splits=splits
    )
    
    return train_dialogs


# Load exact same assistant system prompt as GRPO training
def load_assistant_system_prompt(config):
    data_cfg = config["data"]
    benchmark = data_cfg.get("benchmark", "SGD")
    domain = data_cfg.get("domain", "sgd")
    prompt_category = data_cfg.get("prompt_category", "vanilla")
    iteration = data_cfg.get("iteration", "latest")
    test_mode = data_cfg.get("test_mode", False)
    
    assistant_prompt = get_assistant_prompt(
        benchmark=benchmark,
        domain=domain,
        prompt_category=prompt_category,
        iteration=iteration,
        test_mode=test_mode
    )
    
    return assistant_prompt


# Build conversation history from SGD ground truth dialog
def build_conversation_history(dialog, max_turns=8):
    conversation = []
    turns = dialog.get("turns", [])
        
    turn_idx = 1
    current_turn = {}
    
    for turn_data in turns:
        if turn_idx > max_turns:
            break
            
        speaker = turn_data.get("speaker", "").upper()
        utterance = turn_data.get("utterance", "").strip()
        
        if not utterance:
            continue
            
        if speaker == "USER":
            if current_turn:
                if "user" in current_turn and "agent" in current_turn:
                    current_turn["turn_idx"] = turn_idx - 1
                    conversation.append(current_turn)
                    turn_idx += 1
            current_turn = {"user": utterance}
        elif speaker == "SYSTEM":
            if "user" in current_turn:
                current_turn["agent"] = utterance
    
    if current_turn and "user" in current_turn and "agent" in current_turn:
        current_turn["turn_idx"] = turn_idx
        conversation.append(current_turn)
    
    return conversation


# Convert SGD services to JSON tool calls
def convert_sgd_service_call_to_json(service_call, schemas, dialog):
    short_method = service_call["method"]
    full_tool_name = None
    services = {frame["service"] for turn in dialog["turns"] for frame in turn["frames"] if "service" in frame}
    for service_name in services:
        if service_name in schemas:
            service_schema = schemas[service_name]
            for intent in service_schema["intents"]:
                if intent["name"] == short_method:
                    full_tool_name = f"{service_name}_{short_method}"
                    break
            if full_tool_name:
                break
    
    tool_json = {
        "tool": full_tool_name,
        "args": service_call.get("parameters", {})
    }
    return json.dumps(tool_json, indent=2)


# Format SGD service results as tables for correct tool result format
def format_sgd_service_results_as_table(service_results):
    if not service_results:
        return "No results found."
    if not isinstance(service_results, list) or not service_results:
        return "No results found."
    all_keys = set()
    for result in service_results:
        if isinstance(result, dict):
            all_keys.update(result.keys())
    if not all_keys:
        return "No results found."
    keys = sorted(list(all_keys))
    header = "| " + " | ".join(keys) + " |"
    separator = "| " + " | ".join(["---"] * len(keys)) + " |"
    rows = []
    for result in service_results:
        if isinstance(result, dict):
            row_values = []
            for key in keys:
                value = result.get(key, "")
                if isinstance(value, str):
                    value = value.strip()
                elif value is None:
                    value = ""
                else:
                    value = str(value)
                row_values.append(value)
            row = "| " + " | ".join(row_values) + " |"
            rows.append(row)
    table_parts = [header, separator] + rows
    if len(rows) > 5:
        table_parts = table_parts[:7]
        table_parts.append(f"\n{len(rows) - 5} more records ...")
    
    return "\n".join(table_parts)


# Create tool call
def build_tool_call_sample(dialog, dialog_id, turn_count, conversation_history, service_call, assistant_prompt, schemas):
    tools = get_tools(benchmark="SGD", datapoint=dialog, schemas=schemas, dynamic=True)
    tool_list_str = serialize_tools_for_prompt(tools, schemas)
    context_history = "\n".join(conversation_history)
    input_text = assistant_prompt.replace("{tool_list}", tool_list_str).replace("{conversation}", context_history)
    target_text = convert_sgd_service_call_to_json(service_call, schemas, dialog)
    
    return {
        "dialogue_id": dialog_id,
        "turn_id": turn_count + 1,
        "sample_type": "tool_call",
        "input_text": input_text,
        "target_text": target_text,
        "sgd_service_call": service_call,
    }


# Create post tool reply
def build_post_tool_reply_sample(dialog_id, turn_count, conversation_history, service_call, service_results, assistant_utterance):
    last_user_message = None
    for line in reversed(conversation_history):
        if line.startswith("user: "):
            last_user_message = line[6:]
            break
    tool_result_system_prompt = (
        "Using the tool result below, write a concise, friendly assistant reply to the user.\n"
        "Requirements:\n"
        "- Include the exact values for any attributes the user requested that appear in the result.\n"
        "- Use canonical formats: dates YYYY-MM-DD, times HH:MM, prices as digits only (no $), booleans 'True'/'False'.\n"
        "- Do not paraphrase slot values; state them verbatim where possible.\n"
        "- If multiple rows were returned, either ask one clarifying question to choose, or pick the first row deterministically and proceed.\n"
        "- Do NOT paste raw tables or mention tools.\n"
        "- While you MUST use the exact value returned by the tool call for a given attribute, make your answer natural and human-like."
    )
    
    tool_result_str = format_sgd_service_results_as_table(service_results)
    input_messages = [
        {"role": "system", "content": tool_result_system_prompt}
    ]
    if last_user_message:
        input_messages.append({"role": "user", "content": last_user_message})
    tool_name = service_call["method"]
    input_messages.append({"role": "function", "name": tool_name, "content": tool_result_str})
    input_text = ""
    for msg in input_messages:
        if msg["role"] == "system":
            input_text += f"{msg['content']}\n\n"
        elif msg["role"] == "user":
            input_text += f"User: {msg['content']}\n\n"
        elif msg["role"] == "function":
            input_text += f"Function {msg['name']} result:\n{msg['content']}\n\n"
    
    return {
        "dialogue_id": dialog_id,
        "turn_id": turn_count + 1,
        "sample_type": "post_tool_reply",
        "input_text": input_text.strip(),
        "target_text": assistant_utterance,
        "sgd_service_call": service_call,
        "sgd_service_results": service_results,
    }


#Build SFT input output samples
def process_sgd_dialogs_iteratively(dialogs, assistant_prompt, schemas, max_turns=8):
    samples = []
    
    for i, dialog in enumerate(dialogs):
        dialog_id = dialog.get("dialogue_id", f"dialog_{i}")
        turns = dialog.get("turns", [])        
        conversation_history = []
        turn_count = 0
        
        for turn_idx, turn_data in enumerate(turns):
            if turn_count >= max_turns:
                    break
                
            speaker = turn_data.get("speaker", "").upper()
            utterance = turn_data.get("utterance", "").strip()
            
            if not utterance:
                continue

            if speaker == "USER":
                conversation_history.append(f"user: {utterance}")
            elif speaker == "SYSTEM":
                context_history = "\n".join(conversation_history)
                frames = turn_data.get("frames", [])
                has_tool_call = (
                    frames and 
                    "service_call" in frames[0] and 
                    "service_results" in frames[0]
                )
                
                if has_tool_call:
                    service_call = frames[0]["service_call"]
                    service_results = frames[0]["service_results"]
                    tool_call_sample = build_tool_call_sample(
                        dialog=dialog,
                        dialog_id=dialog_id,
                        turn_count=turn_count,
                        conversation_history=conversation_history,
                        service_call=service_call,
                        assistant_prompt=assistant_prompt,
                        schemas=schemas
                    )
                    samples.append(tool_call_sample)
                    
                    post_tool_sample = build_post_tool_reply_sample(
                        dialog_id=dialog_id,
                        turn_count=turn_count,
                        conversation_history=conversation_history,
                        service_call=service_call,
                        service_results=service_results,
                        assistant_utterance=utterance
                    )
                    samples.append(post_tool_sample)
                    
                else:
                    tools = get_tools(benchmark="SGD", datapoint=dialog, schemas=schemas, dynamic=True)
                    tool_list_str = serialize_tools_for_prompt(tools, schemas)
                    context_history = "\n".join(conversation_history)
                    input_text = assistant_prompt.replace("{tool_list}", tool_list_str).replace("{conversation}", context_history)
                    
                    direct_sample = {
                        "dialogue_id": dialog_id,
                        "turn_id": turn_count + 1,
                    "sample_type": "direct",
                    "input_text": input_text,
                        "target_text": utterance,
                    }
                    samples.append(direct_sample)
                
                conversation_history.append(f"assistant: {utterance}")
                turn_count += 1
        
        if turn_count > 0:
    
    return samples


# Build SFT SGD data set
def main():
    config = load_yaml_config(CONFIG_PATH)
    train_dialogs = load_exact_same_sgd_data(config)
    assistant_prompt = load_assistant_system_prompt(config)
    data_cfg = config["data"]
    schemas = load_schemas()
    grpo_cfg = config.get("grpo", {})
    max_turns = int(grpo_cfg.get("max_turns", 8))
    
    samples = process_sgd_dialogs_iteratively(train_dialogs, assistant_prompt, schemas, max_turns)
    tool_call_samples = [s for s in samples if s["sample_type"] == "tool_call"]
    post_tool_samples = [s for s in samples if s["sample_type"] == "post_tool_reply"]
    direct_samples = [s for s in samples if s["sample_type"] == "direct"]
    
    print(f"Generated {len(samples)} samples: {len(tool_call_samples)} tool calls, {len(post_tool_samples)} post-tool replies, {len(direct_samples)} direct replies")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for sample in samples:
            clean_sample = {k: v for k, v in sample.items() if not k.startswith("sgd_")}
            f.write(json.dumps(clean_sample, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(samples)} samples to {OUTPUT_JSONL}")
    
    return samples

if __name__ == "__main__":
    dialogs = main()
