# Disclaimer: This file consists entirely of code from the AutoTOD codebase
###########################################################################

import json
import re
import sqlite3
import openai
import tenacity
from collections import OrderedDict
from termcolor import cprint, colored
from .sgd_data_utils import prepare_goals_str, INFO_DB_PATH, TRANS_DB_PATH, load_schemas

# Configuration
schemas = load_schemas()

# Extract JSON object from text
def _extract_json_object(text: str):
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text,
                  flags=re.IGNORECASE | re.MULTILINE)
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = text[start:i+1]
                    try:
                        return json.loads(cand)
                    except Exception:
                        break
        start = text.find("{", start + 1)
    raise ValueError("Evaluator did not return valid JSON.")


# Not used
OPENAI_PRICE = {
    'gpt-3.5-turbo': {
        'input': 0.0015 / 1000,
        'output': 0.002 / 1000    
    },
    'gpt-4': {
        'input': 0.03 / 1000,     
        'output': 0.06 / 1000
    },
    'text-davinci': {
        'text': 0.02 / 1000        
    }
}

# Log retry attempts
def tenacity_retry_log(retry_state):
    t = retry_state.next_action.sleep
    e = retry_state.outcome.exception()
    msg = f'Tenacity: Retrying call Agent in {t:.2f} seconds as it raise {e.__class__.__name__}: '
    msg = colored(msg, 'red', force_color=True) + str(e)
    print(msg)

# Not used
def calc_openai_cost(model, usage):
    if model.startswith('gpt-3.5-turbo'):
        price = OPENAI_PRICE['gpt-3.5-turbo']
        cost = usage['prompt_tokens'] * price['input'] + usage['completion_tokens'] * price['output']
    elif model.startswith('gpt-4'):
        price = OPENAI_PRICE['gpt-4']
        cost = usage['prompt_tokens'] * price['input'] + usage['completion_tokens'] * price['output']
    elif model.startswith('text-davinci'):
        price = OPENAI_PRICE['text-davinci']
        cost = usage['total_tokens'] * price['text']
    else:
        raise ValueError(f'{model = }')
    return cost


# Extract user goals from data point with inform and request actions
def extract_user_goals_canonical(dialog):
    goals = OrderedDict()

    for turn in dialog['turns']:
        if turn['speaker'] != 'USER':
            continue
        for frame in turn['frames']:
            service_name = frame['service']
            intent = frame['state']['active_intent']
            if service_name not in goals:
                goals[service_name] = OrderedDict()
            if intent not in goals[service_name]:
                goals[service_name][intent] = {'inform': {}, 'request': []}

            for action in frame['actions']:
                if action['act'] == 'INFORM':
                    goals[service_name][intent]['inform'][action['slot']] = action['canonical_values'][0]
                elif action['act'] == 'REQUEST':
                    goals[service_name][intent]['request'].append(action['slot'])

    goals = {k: v for k, v in goals.items() if v}
    for service_name, service in goals.items():
        service = {k: v for k, v in service.items() if v['inform'] or v['request']}
        for intent in service.values():
            intent['request'] = list(set(intent['request']))
        goals[service_name] = service

    return goals


# Check if inform requirements are satisfied by tool call
def check_inform(service_name, intent_name, inform_state, callings):
    target_func_name = f"{service_name}_{intent_name}".lower()
    for call in callings:
        if call['name'].lower() == target_func_name:
            matches = []
            for k, v in inform_state.items():
                v = str(v).lower()
                vv = call['args'].get(k)
                vv = str(vv).lower() if vv is not None else ""
                matches.append(v == vv)
            if all(matches):
                return True
    return False


# Evaluate inform score for all intents in data point
def evaluate_inform(dialog, callings):
    gold_goals = extract_user_goals_canonical(dialog)

    result = {}
    for service_name, service in gold_goals.items():
        result[service_name] = {}
        for intent_name, intent in service.items():
            inform = check_inform(service_name, intent_name, intent['inform'], callings)
            result[service_name][intent_name] = int(inform)

    return result


# Convert conversation into format for LLM evaluation
def prepare_log_dialog_str(logs):
    dialog_str = []
    for turn in logs:
        dialog_str.append(f"User: {turn['user']}")
        dialog_str.append(f"AI Assistant: {turn['agent']}")
    dialog_str = '\n'.join(dialog_str)
    return dialog_str


# Generate evaluation questions give to LLM-as-a-judge
def prepare_questions_and_answer_formarts(goals):
    questions = []
    answer_formats = []
    q_idx = 1

    for service_name, service in goals.items():
        for intent_name, intent in service.items():
            intent_dict = {it['name']: it for it in schemas[service_name]['intents']}
            slot_dict = {it['name']: it for it in schemas[service_name]['slots']}
            for slot in intent['request']:
                intent_desc = intent_dict[intent_name]['description']
                intent_desc = intent_desc[0].lower() + intent_desc[1:]

                slot_desc = slot_dict[slot]['description']
                slot_desc = slot_desc[0].lower() + slot_desc[1:]

                questions.append(f'{q_idx}. When the user {intent_desc}, what is the {slot_desc}?')
                answer_formats.append(f'"{service_name} {slot}": "<fill the answer of question {q_idx}>"')
                q_idx += 1

    questions = '\n'.join(questions)
    answer_formats = [' ' * 4 + s for s in answer_formats]
    answer_formats = '\n'.join(answer_formats)
    answer_formats = ANSWER_FORMAT_TEMPLATE.format(answer_formats=answer_formats)

    return questions, answer_formats


# Run LLM-as-a-judge to answer the questions
@tenacity.retry(
    wait=tenacity.wait_exponential(min=2, max=30),    
    stop=tenacity.stop_after_attempt(4),              
    reraise=True,
    before_sleep=tenacity_retry_log,
    retry=tenacity.retry_if_exception_type((json.JSONDecodeError, ValueError))
)
def request_slots_llm_qa(goals_str, dialog_str, questions, answer_formats, model):
    human_prompt = HUMAN_TEMPLATE.format(
        goals=goals_str,
        dialog=dialog_str,
        questions=questions,
        answer_formats=answer_formats,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": human_prompt},
    ]
    result_origin, usage = model.chat(messages, return_usage=True)

    provider = getattr(model, "provider", "")
    cost = calc_openai_cost(model.model_name, usage) if provider == "openai" else 0.0

    llm_answer = _extract_json_object(result_origin)

    cleaned = {}
    for k, v in llm_answer.items():
        if isinstance(v, str) and k.endswith("price") and v.startswith("$"):
            cleaned[k] = v.lstrip("$")
        else:
            cleaned[k] = v

    return cleaned, cost


# Extract tool results by querying SGD database
def sgd_function_info(service_name, intent, args, db_path=INFO_DB_PATH):
    is_transactional = intent.get("is_transactional", False)
    
    if is_transactional:
        fields = ', '.join(f'"{field}"' for field in intent['result_slots'])
        sql = f'SELECT {fields} FROM {service_name}'
    else:
        fields = ', '.join(f'"{field}"' for field in intent['result_slots'])
        sql = f'SELECT {fields} FROM {service_name}'
 
    try:
        conn = sqlite3.connect(db_path)
        
        if args:
            conditions = ' AND '.join(f'"{k}" = ?' for k in args.keys())
            sql += f' WHERE {conditions}'
            cursor = conn.execute(sql, list(args.values()))
        else:
            cursor = conn.execute(sql)

        slots = [desc[0] for desc in cursor.description]
        records = []
        for item in cursor:
            record = {slot: value for slot, value in zip(slots, item)}
            records.append(record)

        conn.close()
        return records
    
    except Exception as e:
        print(f"Database error in sgd_function_info: {e}")
        print(f"SQL: {sql}")
        print(f"Args: {args}")
        if 'conn' in locals():
            conn.close()
        return []  


# Compare extracted data base records with extracted answer from the LLM-as-a-judge
def record_satisfying(record, slot_values):
    for k, v in slot_values.items():
        v = str(v).lower()
        vv = record.get(k)
        vv = str(vv).lower()
        if v != vv:
            return False
    return True

def check_success(service_name, intent, callings, slot_values):
    
    normalized_slot_values = {
        k.lower(): str(v).lower() if v is not None else "" for k, v in slot_values.items()
    }

    for call in callings:
        if call['name'].lower().startswith(service_name.lower()):
            normalized_args = {
                k.lower(): str(v).lower() if v is not None else "" for k, v in call['args'].items()
            }

            records = sgd_function_info(service_name, intent, normalized_args)
            
            if records == [] and normalized_args:
                try:
                    test_records = sgd_function_info(service_name, intent, {})
                    if test_records == []:
                        print(f"Returned {len(records)} records")
                    else:
                        print(f"Database error: Query with parameters failed but database has data")
                        print(f"Treating database error as failure (0) since user made requests")
                        return False  
                except Exception as e:
                    print(f"Database error: Exception during database verification: {e}")
                    print(f"Treating database error as failure (0) since user made requests")
                    return False  
            
            print(f"Returned {len(records)} records")
            for r in records:
                print(f"- {r}")

            for record in records:
                record_lc = {k.lower(): str(v).lower() if v is not None else "" for k, v in record.items()}
                if all(
                    normalize_eval_value(normalized_slot_values.get(k)) ==
                    normalize_eval_value(record_lc.get(k))
                    for k in normalized_slot_values
                ):
                    print("Match found: success")
                    return True

            print("No matching record found.")

    return False

# Normalize extracted values for comparison
def normalize_eval_value(v):
    if v is None:
        return ""
    v = str(v).strip().lower()
    v = v.lstrip('$').replace(',', '')
    return v


# Create evaluation result for request slots using LLM-as-a-judge answers
def make_request_eval_result(llm_answer, gold_goals, callings):
    for k, v in llm_answer.items():
        print(f'{k}: {v}')
    print()
    result = {}
    for service_name, service in gold_goals.items():
        result[service_name] = {}
        for intent_name, intent_goals in service.items():
            if intent_goals['request'] == []:
                result[service_name][intent_name] = None
                continue
            slot_values = {slot: llm_answer.get(f'{service_name} {slot}') for slot in intent_goals['request']}
            intent_dict = {it['name']: it for it in schemas[service_name]['intents']}
            intent = intent_dict[intent_name]
            success = check_success(service_name, intent, callings, slot_values)
            result[service_name][intent_name] = int(success)
    return result

# Evaluate request slot satisfaction using LLM-as-a-judge
def evaluate_request(dialog, logs, callings, model):
    goals_str = prepare_goals_str(dialog)

    dialog_str = prepare_log_dialog_str(logs)

    gold_goals = extract_user_goals_canonical(dialog)
    questions, answer_formats = prepare_questions_and_answer_formarts(gold_goals)

    llm_answer, cost = request_slots_llm_qa(goals_str, dialog_str, questions, answer_formats, model)

    result = make_request_eval_result(llm_answer, gold_goals, callings)

    return result, cost

# Main evaluation function combining inform and success
def evaluate(dialog, logs, callings, model):

    inform_result = evaluate_inform(dialog, callings)
    success_result, cost = evaluate_request(dialog, logs, callings, model)

    total_intents = 0
    inform_pass = 0
    success_available = 0
    success_pass = 0
    success_none = 0

    for service_name, intents in inform_result.items():
        for intent_name, inform_val in intents.items():
            total_intents += 1
            inform_pass += int(bool(inform_val))
            succ_val = success_result[service_name][intent_name]
            if succ_val is None:
                success_none += 1
            else:
                success_available += 1
                success_pass += int(bool(succ_val))

    print(f"Inform: {inform_pass}/{total_intents} intents passed | "
        f"Success (pre-combine): {success_pass}/{success_available} passed, {success_none} N/A | Cost: {cost:.4f}")

    eval_result = {}
    gold_goals = extract_user_goals_canonical(dialog)
    for service_name, service_goal in gold_goals.items():
        eval_result[service_name] = {intent_name: {'inform': None, 'success': None} for intent_name in service_goal}

    final_success_total = 0
    final_success_pass = 0

    for service_name, service_result in eval_result.items():
        for intent_name, intent_result in service_result.items():
            inform = inform_result[service_name][intent_name]
            success = success_result[service_name][intent_name]
            intent_result['inform'] = int(inform)
            intent_result['success'] = success if success is None else int(inform and success)
            if intent_result['success'] is not None:
                final_success_total += 1
                final_success_pass += int(bool(intent_result['success']))

    print(f"Final combined success: {final_success_pass}/{final_success_total} (ignoring N/A).")

    return eval_result, cost

# Display evaluation results
def show_eval_result(eval_result):
    for service_name, service_result in eval_result.items():
        print('service: ', end='')
        cprint(service_name, 'red', attrs=['bold'], force_color=True)
        for intent_name, intent_result in service_result.items():
            print(f'intent: ', end='')
            cprint(intent_name, 'green', attrs=['bold'], force_color=True, end='')
            print(intent_result)

# Prompt templates
SYSTEM_PROMPT = '''You are a calm, objective and professional judger and good at to evaluate quality of dialuges between user and AI Assistant. Your judging results are always accurate and concise.'''

HUMAN_TEMPLATE = '''There is a dialogue between a user and an AI Assistant. The user has the goals in his minds (User Goals) and talks with the AI Assistant to achieve the goals. The AI Assistant is a intelligent agent that is able to understand the user utterances, decide to take actions to use external tools, and generate proper responses. Your task is to judge whether the AI Assistant helps the user achieve his goals successfully by answering the questions one by one.

User Goals:

{goals}

Dialogue:

{dialog}

Questions:

{questions}

{answer_formats}'''


ANSWER_FORMAT_TEMPLATE = '''Answer Format:

STRICTLY ONLY output the answer in a JSON format this. Do not add any other text:
{{
{answer_formats}
}}
Only output the JSON format, no other text.
'''