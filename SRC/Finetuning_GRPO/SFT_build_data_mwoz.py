import json
import random
import sys
from pathlib import Path
sys.path.append("SRC")
from Modular_Learning_Agent.utils_accessing import get_assistant_prompt
from Modular_Learning_Agent.utils_tools import get_tools
from Modular_Learning_Agent.utils_sim import serialize_tools_for_prompt
from Finetuning_GRPO.MTRA_utils_config import load_yaml_config
from Modular_Learning_Agent.multiwoz_data_utils import load_data_split

# Configurations
CONFIG_PATH = "SRC/Finetuning_GRPO/MTRA_config.yaml"
OUTPUT_JSONL = "MWOZ_SFT_Data.jsonl"

DOMAIN_TOOL_MAPPING = {
    "restaurant": {
        "query": "query_restaurants",
        "book": "book_restaurant"
    },
    "hotel": {
        "query": "query_hotels", 
        "book": "book_hotel"
    },
    "attraction": {
        "query": "query_attractions",
        "book": None 
    },
    "train": {
        "query": "query_trains",
        "book": "buy_train_tickets"
    },
    "taxi": {
        "query": None, 
        "book": "book_taxi"
    }
}

# Load MultiWOZ dialogs using same selection logic as GRPO training
def load_mwoz_data(n_dialogs=500, seed=42):
    data = load_data_split("train")
    dialog_ids = sorted(list(data.keys()))
    random.seed(seed)
    if n_dialogs < len(dialog_ids):
        dialog_ids = random.sample(dialog_ids, n_dialogs)
    dialogs = []
    for dialog_id in dialog_ids:
        dialog = data[dialog_id]
        dialog["dialogue_id"] = dialog_id
        dialogs.append(dialog)
    
    return dialogs


# Load exact same assistant system prompt as GRPO training
def load_assistant_system_prompt(config):
    data_cfg = config["data"]
    benchmark = "MultiWOZ"
    domain = data_cfg.get("domain", "mwoz")
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


# Convert MultiWOZ belief state constraints to SQL query
def convert_constraints_to_sql(domain, constraints):
    where_clauses = []
    for key, value in constraints.items():
        if value and value != "" and value != "not mentioned":
            if key == "pricerange":
                key = "pricerange"
            elif key == "food":
                key = "food"
            elif key == "type":
                key = "type"
            where_clauses.append(f"{key} = '{value}'")
    if where_clauses:
        where_str = " AND ".join(where_clauses)
        sql = f"SELECT * FROM {domain} WHERE {where_str}"
    else:
        sql = f"SELECT * FROM {domain}"
    
    return sql


# Detect tool calls from MultiWOZ belief state changes
def detect_tool_call_from_belief_state(current_metadata, previous_metadata, turn_text):
    for domain, domain_data in current_metadata.items():
        if domain not in DOMAIN_TOOL_MAPPING or not isinstance(domain_data, dict):
            continue
            
        book_data = domain_data.get("book", {})
        booked = book_data.get("booked", [])
        
        if booked and DOMAIN_TOOL_MAPPING[domain]["book"]:
            tool_name = DOMAIN_TOOL_MAPPING[domain]["book"]
            booking = booked[0] if booked else {}
            if domain == "restaurant":
                tool_args = {
                    "name": booking.get("name", ""),
                    "people": int(book_data.get("people", "1")) if book_data.get("people", "").isdigit() else 1,
                    "day": book_data.get("day", ""),
                    "time": book_data.get("time", "")
                }
            elif domain == "hotel":
                tool_args = {
                    "name": booking.get("name", ""),
                    "people": int(book_data.get("people", "1")) if book_data.get("people", "").isdigit() else 1,
                    "day": book_data.get("day", ""),
                    "stay": int(book_data.get("stay", "1")) if book_data.get("stay", "").isdigit() else 1
                }
            elif domain == "train":
                tool_args = {
                    "train_id": booking.get("trainID", ""),
                    "tickets": int(book_data.get("people", "1")) if book_data.get("people", "").isdigit() else 1
                }
            elif domain == "taxi":
                semi_data = domain_data.get("semi", {})
                tool_args = {
                    "departure": semi_data.get("departure", ""),
                    "destination": semi_data.get("destination", "")
                }
                if semi_data.get("leaveAt"):
                    tool_args["leave_time"] = semi_data.get("leaveAt")
                if semi_data.get("arriveBy"):
                    tool_args["arrive_time"] = semi_data.get("arriveBy")
            tool_results = f"Booking succeed. The reference number is {booking.get('reference', 'ABC12345')}."
            
            return True, tool_name, tool_args, tool_results
        
        semi_data = domain_data.get("semi", {})
        prev_domain_data = previous_metadata.get(domain, {})
        prev_semi_data = prev_domain_data.get("semi", {}) if isinstance(prev_domain_data, dict) else {}
        current_constraints = {k: v for k, v in semi_data.items() 
                             if v and v != "" and v != "not mentioned"}
        previous_constraints = {k: v for k, v in prev_semi_data.items() 
                              if v and v != "" and v != "not mentioned"}
        if current_constraints != previous_constraints and current_constraints and DOMAIN_TOOL_MAPPING[domain]["query"]:
            tool_name = DOMAIN_TOOL_MAPPING[domain]["query"]
            sql_query = convert_constraints_to_sql(domain, current_constraints)
            tool_args = {"sql": sql_query}
            tool_results = f"| name | area | phone |\n| Example {domain.title()} | centre | 01234567890 |\n1 more records ..."
            
            return True, tool_name, tool_args, tool_results
    
    return False, "", {}, ""


# Convert MultiWOZ tool call to JSON format
def convert_to_json_tool_call(tool_name, tool_args):
    tool_json = {
        "tool": tool_name,
        "args": tool_args
    }
    return json.dumps(tool_json, indent=2)


# Format MultiWOZ tool results to match actual tool output format
def format_mwoz_tool_results_as_table(tool_results):
    if isinstance(tool_results, str):
        return tool_results
    elif not tool_results:
        return "No results found."
    else:
        return str(tool_results)


# Create tool call json
def build_mwoz_tool_call_sample(dialog, dialog_id, turn_count, conversation_history, tool_name, tool_args, assistant_prompt):
    tools = get_tools(benchmark="MultiWOZ", datapoint=dialog, dynamic=True)
    tool_list_str = serialize_tools_for_prompt(tools)
    context_history = "\n".join(conversation_history)
    input_text = assistant_prompt.replace("{tool_list}", tool_list_str).replace("{conversation}", context_history)
    target_text = convert_to_json_tool_call(tool_name, tool_args)
    
    return {
        "dialogue_id": dialog_id,
        "turn_id": turn_count + 1,
        "sample_type": "tool_call",
        "input_text": input_text,
        "target_text": target_text,
    }


# Create post tool reply
def build_mwoz_post_tool_reply_sample(dialog_id, turn_count, conversation_history, tool_name, tool_results, assistant_utterance):
    last_user_message = None
    for line in reversed(conversation_history):
        if line.startswith("User: "):
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
    
    input_messages = [
        {"role": "system", "content": tool_result_system_prompt}
    ]
    if last_user_message:
        input_messages.append({"role": "user", "content": last_user_message})
    tool_result_table = format_mwoz_tool_results_as_table(tool_results)
    input_messages.append({
        "role": "function", 
        "name": tool_name,
        "content": tool_result_table
    })
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
    }


# Create direct reply
def build_mwoz_direct_reply_sample(dialog, dialog_id, turn_count, conversation_history, assistant_utterance, assistant_prompt):
    tools = get_tools(benchmark="MultiWOZ", datapoint=dialog, dynamic=True)
    tool_list_str = serialize_tools_for_prompt(tools)
    context_history = "\n".join(conversation_history)
    input_text = assistant_prompt.replace("{tool_list}", tool_list_str).replace("{conversation}", context_history)
    
    return {
        "dialogue_id": dialog_id,
        "turn_id": turn_count + 1,
        "sample_type": "direct",
        "input_text": input_text,
        "target_text": assistant_utterance,
    }


# Build conversation history from MultiWOZ dialog
def build_conversation_history(dialog, max_turns=8):
    conversation = []
    
    log = dialog.get("log", [])
    turn_count = 0
    
    for i in range(0, len(log), 2):
        if turn_count >= max_turns:
            break
        if i < len(log):
            user_text = log[i].get("text", "")
            conversation.append(f"User: {user_text}")
        if i + 1 < len(log):
            system_text = log[i + 1].get("text", "")
            conversation.append(f"Assistant: {system_text}")
            turn_count += 1
    
    return conversation


# Build SFT samples
def process_mwoz_dialogs_iteratively(dialogs, assistant_prompt, max_turns=8):
    samples = []
    
    for i, dialog in enumerate(dialogs):
        dialog_id = dialog.get("dialogue_id", f"dialog_{i}")
        log = dialog.get("log", [])
        
        conversation_history = []
        turn_count = 0
        
        for j in range(1, len(log), 2):
            if turn_count >= max_turns:
                break
            if j - 1 < len(log):
                user_text = log[j - 1].get("text", "")
                conversation_history.append(f"User: {user_text}")
            system_turn = log[j]
            system_text = system_turn.get("text", "")
            current_metadata = system_turn.get("metadata", {})
            previous_metadata = log[j - 2].get("metadata", {}) if j >= 2 else {}
            has_tool_call, tool_name, tool_args, tool_results = detect_tool_call_from_belief_state(
                current_metadata, previous_metadata, system_text
            )
            
            if has_tool_call:
                
                tool_call_sample = build_mwoz_tool_call_sample(
                    dialog=dialog,
                    dialog_id=dialog_id,
                    turn_count=turn_count,
                    conversation_history=conversation_history,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    assistant_prompt=assistant_prompt
                )
                samples.append(tool_call_sample)
                
                post_tool_sample = build_mwoz_post_tool_reply_sample(
                    dialog_id=dialog_id,
                    turn_count=turn_count,
                    conversation_history=conversation_history,
                    tool_name=tool_name,
                    tool_results=tool_results,
                    assistant_utterance=system_text
                )
                samples.append(post_tool_sample)
                
            else:
                
                direct_sample = build_mwoz_direct_reply_sample(
                    dialog=dialog,
                    dialog_id=dialog_id,
                    turn_count=turn_count,
                    conversation_history=conversation_history,
                    assistant_utterance=system_text,
                    assistant_prompt=assistant_prompt
                )
                samples.append(direct_sample)
                
            conversation_history.append(f"Assistant: {system_text}")
            turn_count += 1

    return samples


# Main function to build MultiWOZ SFT dataset
def main():
    config = load_yaml_config(CONFIG_PATH)
    data_cfg = config.get("data", {})
    n_datapoints = int(data_cfg.get("n_datapoints", 500))
    data_seed = int(data_cfg.get("data_seed", 42))
    
    dialogs = load_mwoz_data(n_datapoints, data_seed)
    assistant_prompt = load_assistant_system_prompt(config)
    grpo_cfg = config.get("grpo", {})
    max_turns = int(grpo_cfg.get("max_turns", 8))
    samples = process_mwoz_dialogs_iteratively(dialogs, assistant_prompt, max_turns)
    
    tool_call_samples = [s for s in samples if s["sample_type"] == "tool_call"]
    post_tool_samples = [s for s in samples if s["sample_type"] == "post_tool_reply"]
    direct_samples = [s for s in samples if s["sample_type"] == "direct"]
    
    print(f"Generated {len(samples)} samples: {len(tool_call_samples)} tool calls, {len(post_tool_samples)} post-tool replies, {len(direct_samples)} direct replies")
    
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for sample in samples:
            clean_sample = {k: v for k, v in sample.items() if not k.startswith("mwoz_")}
            f.write(json.dumps(clean_sample, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(samples)} samples to {OUTPUT_JSONL}")
    
    return samples


if __name__ == "__main__":
    main()