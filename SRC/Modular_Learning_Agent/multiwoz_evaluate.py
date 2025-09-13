# Disclaimer: This file consists entirely of code from the AutoTOD codebase
###########################################################################

import json
import re
from pprint import pprint
import openai
import tenacity
from .multiwoz_tools import query_trains, query_train_by_id, query_venue_by_name, query_booking_by_refer_num, Venue, BookRecord
from .multiwoz_data_utils import DOMAINS, clean_time, prepare_goals_string, tenacity_retry_log


# Prompts for LLM as a judge
SYSTEM_PROMPT = '''You are a calm, objective and professional evaluator. Your task is to evaluate dialogue quality between user and AI Assistant. 

CRITICAL: You MUST respond with valid JSON only. Do not include any explanatory text, markdown formatting, or additional content. Return only the raw JSON object as specified in the answer format.'''

HUMAN_TEMPLATE = '''There is a dialogue between a user and an AI Assistant. The user has the goals in his minds (User Goals) and talks with the AI Assistant to achieve the goals. The AI Assistant is a intelligent agent that is able to understand the user utterances, decide to take actions to use external tools, and generate proper responses. Your task is to judge whether the AI Assistant helps the user achieve his goals successfully by answering the questions one by one.

User Goals:

{goals}

Dialogue:

{dialog}

Questions:

{questions}

{answer_formats}

IMPORTANT: Respond with ONLY the JSON object. No explanations, no markdown, no additional text.'''


ANSWER_FORMAT_TEMPLATE = '''REQUIRED JSON FORMAT:

You must respond with EXACTLY this JSON structure (no markdown, no backticks, no extra text):

{{
{answer_formats}
}}

EXAMPLE: If asked about a hotel, respond like this:
{{"hotel": "Cambridge Lodge", "reference number": "ABC123"}}

RULES:
- Return ONLY the JSON object above
- If no answer for a question, use "none" as the value  
- Do not include any text before or after the JSON
- Do not wrap in markdown code blocks
- Ensure valid JSON syntax'''

# Convert dialog intro needed format for evaluation
def prepare_dialog_string_with_action(dialog):
    dialog_str = []
    for turn in dialog:
        turn_str = []
        turn_str.append(f'Turn {turn["turn_idx"]}:')
        turn_str.append(f'User: {turn["user"]}')
        turn_str.append(f'AI Assistant: {turn["agent"]}')
        if len(turn['actions']) > 0:
            for action in turn['actions']:
                turn_str.append(f'System Action: {action["action_name"]}; Action Input: {action["action_input"]}')
        else:
            turn_str.append('System Action: None')
        turn_str = '\n'.join(turn_str)
        dialog_str.append(turn_str)
    dialog_str = '\n\n'.join(dialog_str)
    return dialog_str


# Convert dialog to simple text
def prepare_dialog_string(dialog):
    dialog_str = []
    for turn in dialog:
        dialog_str.append(f'User: {turn["user"]}')
        dialog_str.append(f'AI Assistant: {turn["agent"]}')
    dialog_str = '\n'.join(dialog_str)
    return dialog_str

# Taxi domain
TAXI_SLOT_MAP = {
    'departure': 'departure',
    'destination': 'destination',
    'leaveAt': 'leave time',
    'arriveBy': 'arrival time',
    'car type': 'car type',
    'phone': 'phone number',
}


# Generate evaluation questions given to the LLM to evaluate the taxi domain
def prepare_taxi_questions(goal):
    questions = []
    answer_formats = []
    q_idx = 1

    for slot in goal['info']:
        slot_mapped = TAXI_SLOT_MAP[slot]
        q = f'{q_idx}. What is the {slot_mapped} of the taxi that the user books?'
        a = f'"{slot_mapped}": "<fill the answer of question {q_idx}>"'
        questions.append(q)
        answer_formats.append(a)
        q_idx += 1

    for slot in goal['reqt']:
        slot_mapped = TAXI_SLOT_MAP[slot]
        q = f'{q_idx}. What is the {slot_mapped} of the taxi?'
        a = f'"{slot_mapped}": "<fill the answer of question {q_idx}>"'
        questions.append(q)
        answer_formats.append(a)
        q_idx += 1

    questions = '\n'.join(questions)

    answer_formats = [' ' * 4 + s for s in answer_formats]
    answer_formats = '\n'.join(answer_formats)
    answer_formats = ANSWER_FORMAT_TEMPLATE.format(answer_formats=answer_formats)

    return questions, answer_formats


# Evaluate taxi domain
def evaluate_by_domain_taxi(goal, llm_answer):
    result = {
        'domain': 'taxi',
        'goal': goal,
        'inform': {
            'complete': None,
            'slot_values': None,
        },
        'success': {
            'complete': None,
            'slot_values': None,
        },
        'book': {
            'complete': None,
        }
    }

    for slot in ['leave time', 'arrival time']:
        if time := llm_answer.get(slot):
            llm_answer[slot] = clean_time(time)

    if goal.get('info'):
        slot_values = {slot: llm_answer[TAXI_SLOT_MAP[slot]] for slot in goal['info']}
        complete = all(v.lower() == slot_values[s].lower() for s, v in goal['info'].items())

        result['inform']['complete'] = int(complete)
        result['inform']['slot_values'] = slot_values

    if goal.get('reqt'):
        slot_values = {slot: llm_answer[TAXI_SLOT_MAP[slot]] for slot in goal['reqt']}
        complete = all(v != 'none' for v in slot_values.values())

        result['success']['complete'] = int(complete and result['inform']['complete'])
        result['success']['slot_values'] = slot_values

    return result

# Train domain
TRAIN_SLOT_MAP = {
    'trainID': 'train id',
    'price': 'price',
    'duration': 'duration',
    'leaveAt': 'leave time',
    'arriveBy': 'arrive time',
}


# Generate evaluation questions given to the LLM to evaluate train domain
def prepare_train_questions(goal):

    questions = []
    answer_formats = []

    if goal.get('book'):
        assert not goal.get('reqt')
        q = f'1. What is the reference number of the booked train tickets?'
        a = f'"reference number": "<fill the answer of question 1>"'
        questions.append(q)
        answer_formats.append(a)
    
    else:  
        assert not goal.get('book')
        q_idx = 1
        for slot in goal.get('reqt'):
            slot_mapped = TRAIN_SLOT_MAP[slot]
            if slot == 'trainID':
                q = f'{q_idx}. What is the id of the train?'
                a = f'"{slot_mapped}": "<fill the answer of question {q_idx}>"'
            else:
                q = f'{q_idx}. What is the {slot_mapped} of the train?'
                a = f'"{slot_mapped}": "<fill the answer of question {q_idx}>"'
            questions.append(q)
            answer_formats.append(a)
            q_idx += 1

    questions = '\n'.join(questions)

    answer_formats = [' ' * 4 + s for s in answer_formats]
    answer_formats = '\n'.join(answer_formats)
    answer_formats = ANSWER_FORMAT_TEMPLATE.format(answer_formats=answer_formats)

    return questions, answer_formats


# Evaluate train domain
def evaluate_by_domain_train(goal, llm_answer):

    result = {
        'domain': 'train',
        'goal': goal,
        'inform': {
            'complete': None,
        },
        'success': {
            'complete': None,
            'slot_values': None,
        },
        'book': {
            'complete': None,
            'refer_number': None,
            'book_record': None,
            'train_info': None,
        }
    }

    for slot in ['leave time', 'arrival time']:
        if time := llm_answer.get(slot):
            llm_answer[slot] = clean_time(time)

    if goal.get('reqt'):
        slot_values = {slot: llm_answer[TRAIN_SLOT_MAP[slot]] for slot in goal['reqt']}
        items = query_trains(goal['info'])
        complete = any(item.satisfying(slot_values) for item in items)

        result['inform']['complete'] = int(complete)
        result['success']['complete'] = int(complete)
        result['success']['slot_values'] = slot_values

    if goal.get('book'):
        refer_number = llm_answer['reference number']
        if book_record := query_booking_by_refer_num('train', refer_number):
            if train := query_train_by_id(book_record.trainID):
                inform_complete = train.satisfying(goal['info'])
                book_complete = inform_complete and book_record.satisfying({'tickets': goal['book']['people']})
            else:
                train = f'"{book_record.trainID}" is not found in the "train" table.'
                inform_complete, book_complete = False, False
        else:
            book_record = f'"{refer_number}" is not found in the "book_train" table.'
            train = f'No train as invalid refer number "{refer_number}".'
            inform_complete, book_complete = False, False

        result['book']['refer_number'] = refer_number
        result['inform']['complete'] = int(inform_complete)
        result['book']['complete'] = int(book_complete)
        result['book']['book_record'] = book_record
        result['book']['train_info'] = train

    return result


# Hotel, restaurant and attraction domains

# Generate evaluation questions given to the LLM to evaluate hotel domain
def prepare_hotel_questions(goal):
    questions = []
    answer_formats = []
    q_idx = 1

    if goal.get('book', []):
        q = f'{q_idx}. What hotel does the user choose and would like to book it?'
    else:
        q = f'{q_idx}. What hotel is the user interested in and asking information about it?'
    a = f'"hotel": "<fill the answer of question {q_idx}>"'
    questions.append(q)
    answer_formats.append(a)
    q_idx += 1

    if goal.get('book'):
        q = f'{q_idx}. What is the reference number of the booked hotel?'
        a = f'"reference number": "<fill the answer of question {q_idx}>"'
        questions.append(q)
        answer_formats.append(a)
        q_idx += 1

    for slot in goal.get('reqt', []):
        if slot == 'area':
            q = f'{q_idx}. What is the area of the hotel? (east, west, north, south, centre)'
            a = f'"{slot}": "<fill the answer of question {q_idx} (east, west, north, south, centre)>"'
        elif slot == 'pricerange':
            q = f'{q_idx}. What is the price of the hotel? (cheap, moderate, expensive)'
            a = f'"{slot}": "<fill the answer of question {q_idx} (cheap, moderate, expensive)>"'
        elif slot == 'type':
            q = f'{q_idx}. What is the type of the hotel? (guesthouse, hotel)'
            a = f'"{slot}": "<fill the answer of question {q_idx} (guesthouse, hotel)>"'
        elif slot == 'stars':
            q = f'{q_idx}. What is the stars of the hotel? (1, 2, 3, ...)'
            a = f'"{slot}": "<fill the answer of question {q_idx} (1, 2, 3, ...)>"'
        elif slot == 'internet':
            q = f'{q_idx}. Does the hotel have free internet/wifi? (yes, no)'
            a = f'"{slot}": "<fill the answer of question {q_idx} (yes, no)>"'
        elif slot == 'parking':
            q = f'{q_idx}.  Does the hotel have free parking? (yes, no)'
            a = f'"{slot}": "<fill the answer of question {q_idx} (yes, no)>"'
        elif slot == 'address':
            q = f'{q_idx}. What is the address of the hotel?'
            a = f'"{slot}": "<fill the answer of question {q_idx}>"'
        elif slot == 'phone':
            q = f'{q_idx}. What is the phone number of the hotel?'
            a = f'"{slot}": "<fill the answer of question {q_idx}>"'
        elif slot == 'postcode':
            q = f'{q_idx}. What is the postcode of the hotel?'
            a = f'"{slot}": "<fill the answer of question {q_idx}>"'
        else:
            q = None
            a = None

        if q and a:
            questions.append(q)
            answer_formats.append(a)
            q_idx += 1

    questions = '\n'.join(questions)

    answer_formats = [' ' * 4 + s for s in answer_formats]
    answer_formats = '\n'.join(answer_formats)
    answer_formats = ANSWER_FORMAT_TEMPLATE.format(answer_formats=answer_formats)

    return questions, answer_formats


# Generate evaluation questions given to the LLM to evaluate restaurant domain
def prepare_restaurant_questions(goal):

    questions = []
    answer_formats = []
    q_idx = 1

    if goal.get('book', []):
        q = f'{q_idx}. What restaurant does the user choose and would like to book it?'
    else:
        q = f'{q_idx}. What restaurant is the user interested in and asking information about it?'
    a = f'"restaurant": "<fill the answer of question {q_idx}>"'
    questions.append(q)
    answer_formats.append(a)
    q_idx += 1

    if goal.get('book'):
        q = f'{q_idx}. What is the reference number of the booked restaurant?'
        a = f'"reference number": "<fill the answer of question {q_idx}>"'
        questions.append(q)
        answer_formats.append(a)
        q_idx += 1

    for slot in goal.get('reqt', []):
        if slot == 'area':
            q = f'{q_idx}. What is the area of the hotel? (east, west, north, south, centre)'
            a = f'"{slot}": "<fill the answer of question {q_idx} (east, west, north, south, centre)>"'
        elif slot == 'pricerange':
            q = f'{q_idx}. What is the price of the hotel? (cheap, moderate, expensive)'
            a = f'"{slot}": "<fill the answer of question {q_idx} (cheap, moderate, expensive)>"'
        elif slot == 'food':
            q = f'{q_idx}. What is the food type of the restaurant?'
            a = f'"{slot}": "<fill the answer of question {q_idx}>"'
        elif slot == 'address':
            q = f'{q_idx}. What is the address of the hotel?'
            a = f'"{slot}": "<fill the answer of question {q_idx}>"'
        elif slot == 'phone':
            q = f'{q_idx}. What is the phone number of the hotel?'
            a = f'"{slot}": "<fill the answer of question {q_idx}>"'
        elif slot == 'postcode':
            q = f'{q_idx}. What is the postcode of the hotel?'
            a = f'"{slot}": "<fill the answer of question {q_idx}>"'
        else:
            q = None
            a = None

        if q and a:
            questions.append(q)
            answer_formats.append(a)
            q_idx += 1

    questions = '\n'.join(questions)

    answer_formats = [' ' * 4 + s for s in answer_formats]
    answer_formats = '\n'.join(answer_formats)
    answer_formats = ANSWER_FORMAT_TEMPLATE.format(answer_formats=answer_formats)

    return questions, answer_formats


# Generate evaluation questions given to the LLM to evaluate attraction domain
def prepare_attraction_questions(goal):

    questions = []
    answer_formats = []
    q_idx = 1

    q = f'{q_idx}. What attraction is the user interested in and asking information about it?'
    a = f'"attraction": "<fill the answer of question {q_idx}>"'
    questions.append(q)
    answer_formats.append(a)
    q_idx += 1

    for slot in goal.get('reqt', []):
        if slot == 'area':
            q = f'{q_idx}. What is the area of the attraction? (east, west, north, south, centre)'
            a = f'"{slot}": "<fill the answer of question {q_idx} (east, west, north, south, centre)>"'
        elif slot == 'entrance fee':
            q = f'{q_idx}. What is the entrance fee of the attraction? (cheap, moderate, expensive)'
            a = f'"{slot}": "<fill the answer of question {q_idx} (cheap, moderate, expensive)>"'
        elif slot == 'type':
            q = f'{q_idx}. What is the type of the attraction? (guesthouse, hotel)'
            a = f'"{slot}": "<fill the answer of question {q_idx} (guesthouse, hotel)>"'
        elif slot == 'address':
            q = f'{q_idx}. What is the address of the attraction?'
            a = f'"{slot}": "<fill the answer of question {q_idx}>"'
        elif slot == 'phone':
            q = f'{q_idx}. What is the phone number of the attraction?'
            a = f'"{slot}": "<fill the answer of question {q_idx}>"'
        elif slot == 'postcode':
            q = f'{q_idx}. What is the postcode of the attraction?'
            a = f'"{slot}": "<fill the answer of question {q_idx}>"'
        else:
            q = None
            a = None

        if q and a:
            questions.append(q)
            answer_formats.append(a)
            q_idx += 1

    questions = '\n'.join(questions)

    answer_formats = [' ' * 4 + s for s in answer_formats]
    answer_formats = '\n'.join(answer_formats)
    answer_formats = ANSWER_FORMAT_TEMPLATE.format(answer_formats=answer_formats)

    return questions, answer_formats


# Evaluate restaurant, hotel and attraction domain
def evaluate_by_domain_others(goal, llm_answer, domain):
    result = {
        'domain': domain,
        'goal': goal,
        'inform': {
            'complete': None,
            'venue_name': None,
            'venue_info': None,
        },
        'success': {
            'complete': None,
            'slot_values': None,
        },
        'book': {
            'complete': None,
            'refer_number': None,
            'book_record': None,
        }
    }

    name = llm_answer[domain]
    venue = query_venue_by_name(domain=domain, name=name)
    if venue:
        complete = venue.satisfying(goal['info']) or \
            bool(goal['fail_info']) and venue.satisfying(goal['fail_info'])
    else:
        venue = f'"{name}" is not found in the "{domain}" table.'
        complete = False
    
    result['inform']['complete'] = int(complete)
    result['inform']['venue_name'] = name
    result['inform']['venue_info'] = venue

    if goal.get('reqt'):
        slot_values = {s: llm_answer[s] for s in goal['reqt']}
        if isinstance(venue, Venue) and result['inform']['complete']:
            complete = venue.satisfying(slot_values)
        else:
            complete = False

        result['success']['slot_values'] = slot_values
        result['success']['complete'] = int(complete)

    if goal.get('book'):
        refer_number = llm_answer['reference number']
        book_record = query_booking_by_refer_num(domain=domain, refer_number=refer_number)
        if book_record is None:
            book_record = f'"{refer_number}" is not found in the "book_{domain}" table.'

        if result['inform']['complete'] and isinstance(book_record, BookRecord):
            f1 = book_record.name == venue.name
            f2 = book_record.satisfying(goal['book']) or \
                bool(goal['fail_book']) and book_record.satisfying(goal['fail_book'])
            complete = f1 and f2
        else:
            complete = False

        result['book']['refer_number'] = refer_number
        result['book']['complete'] = int(complete)
        result['book']['book_record'] = book_record

    return result


# Query LLM with previously defined questions
@tenacity.retry(wait=tenacity.wait_exponential(min=1, max=8),
                stop=tenacity.stop_after_attempt(3),
                before_sleep=tenacity_retry_log,
                retry=tenacity.retry_if_exception_type((openai.OpenAIError, json.JSONDecodeError)))
def llm_qa(goal_messages, dialog_pred, questions, answer_formats, model):
    goals_str = prepare_goals_string(goal_messages)
    dialog_str = prepare_dialog_string(dialog_pred)
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
    cost = 0.0

    # Cleaning json string
    def clean_json_string(text):
        text = text.strip()
        text = text.strip('`')
        if text.startswith('```json'):
            text = text[7:]
        elif text.startswith('```'):
            text = text[3:]
        if (pos := text.find('```')) > -1:
            text = text[:pos]
        text = text.strip()
        return text
    
    # Parsing JSON with fallbacks
    def try_parse_json_with_fallback(text, questions, answer_formats):
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            text_lower = text.lower().strip()
            if any(word in text_lower for word in ['none', 'no', 'not mentioned', 'not found', 'empty']):
                fallback_json = {}
                for line in answer_formats.split('\n'):
                    if '":' in line and '"' in line:
                        key_match = re.search(r'"([^"]+)":', line)
                        if key_match:
                            fallback_json[key_match.group(1)] = "none"
                return fallback_json
            
            return {"error": "Could not parse response", "raw_response": text[:100]}
    
    result_cleann = clean_json_string(result_origin)
    llm_answer = try_parse_json_with_fallback(result_cleann, questions, answer_formats)
    return llm_answer, cost


# Display evaluation results
def show_eval_result(result):
    RED = '\u001b[1;31m'
    GREEN = '\u001b[1;33m'
    RESET = '\u001b[0m'
    for k, v in result.items():
        if isinstance(v, str) or isinstance(v, float) or v is None:
            print(RED + f'[{k}]' + RESET + f' {v}')

        elif isinstance(v, dict):
            indent = 4
            if 'complete' in v:
                print(RED + f'[{k}] complete: {v["complete"]}' + RESET)
            else:
                print(RED + f'[{k}]' + RESET)
            for kk, vv in v.items():
                if kk == 'complete':
                    continue
                print(' ' * indent + GREEN + f'{kk}: ' + RESET, end='')
                print(vv)

        else:
            print(RED + f'[{k}]' + RESET)
            pprint(v)


# Main evaluation function routing to the domain specific evaluators
def evaluate_by_domain(domain, run_result, model, verbose=True):
    assert run_result['goals'].get(domain)
    
    goal_dict = run_result['goals'][domain]
    dialog_pred = run_result['dialog_pred']
    goal_messages = run_result['goal_messages']

    if domain == 'hotel':
        questions, answer_formats = prepare_hotel_questions(goal_dict)
        llm_answer, cost = llm_qa(goal_messages, dialog_pred, questions, answer_formats, model)
        result = evaluate_by_domain_others(goal_dict, llm_answer, domain)

    elif domain == 'restaurant':
        questions, answer_formats = prepare_restaurant_questions(goal_dict)
        llm_answer, cost = llm_qa(goal_messages, dialog_pred, questions, answer_formats, model)
        result = evaluate_by_domain_others(goal_dict, llm_answer, domain)

    elif domain == 'attraction':
        questions, answer_formats = prepare_attraction_questions(goal_dict)
        llm_answer, cost = llm_qa(goal_messages, dialog_pred, questions, answer_formats, model)
        result = evaluate_by_domain_others(goal_dict, llm_answer, domain)

    elif domain == 'train':
        questions, answer_formats = prepare_train_questions(goal_dict)
        llm_answer, cost = llm_qa(goal_messages, dialog_pred, questions, answer_formats, model)
        result = evaluate_by_domain_train(goal_dict, llm_answer)

    elif domain == 'taxi':
        questions, answer_formats = prepare_taxi_questions(goal_dict)
        llm_answer, cost = llm_qa(goal_messages, dialog_pred, questions, answer_formats, model)
        result = evaluate_by_domain_taxi(goal_dict, llm_answer)

    else:
        raise ValueError(f'{domain = }')
    
    if verbose:
        show_eval_result(result)

    result['cost'] = cost
    return result