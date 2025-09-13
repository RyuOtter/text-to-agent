# Disclaimer: This file consists entirely of code from the AutoTOD codebase
###########################################################################

import json
import os
import random
from pathlib import Path
from termcolor import cprint
from collections import OrderedDict

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "Data" / "sgd" / "origin"
INFO_DB_PATH = PROJECT_ROOT / "Data" / "sgd" / "db" / "sgd.db"
TRANS_DB_PATH = PROJECT_ROOT / "Data" / "sgd" / "db" / "sgd_trans.db"

schemas = None
dialogs = None

# Loads the blueprints called schemas needed for SGD tools
def load_schemas(data_dir=DATA_DIR):
    global schemas

    if schemas is not None:
        return schemas

    schema_list = []
    with open(os.path.join(data_dir, 'train', 'schema.json')) as f:
        schema_list += json.load(f)
    with open(os.path.join(data_dir, 'dev', 'schema.json')) as f:
        schema_list += json.load(f)
    with open(os.path.join(data_dir, 'test', 'schema.json')) as f:
        schema_list += json.load(f)

    schemas = {schema['service_name']: schema for schema in schema_list}
    return schemas

# Loads SGD data
def load_dialogs(data_dir=DATA_DIR):
    global dialogs

    if dialogs is not None:
        return dialogs
    
    # Load specific split of SGD data
    def load_dialogs_split(split):
        dialogs = []
        data_sub_dir = os.path.join(data_dir, split)
        print(f'Loading dialogs from "{data_sub_dir}"...')
        for name in os.listdir(data_sub_dir):
            if name.startswith('dialogues_'):
                with open(os.path.join(data_sub_dir, name)) as f:
                    dialogs += json.load(f)
        for dialog in dialogs:
            dialog['dialogue_id'] = split + '_' + dialog['dialogue_id']
        return dialogs

    dialogs = []
    dialogs += load_dialogs_split('train')
    dialogs += load_dialogs_split('dev')
    dialogs += load_dialogs_split('test')

    dialogs = {d['dialogue_id']: d for d in dialogs}

    print(f'Loading completed. {len(dialogs)} dialogs loaded.')
    return dialogs

# Select a conversation from the SGD data
def pick_dialog(dialogs, dialog_id='random'):
    if dialog_id == 'random':
        dialog_id = random.choice(list(dialogs.keys()))
    else:
        assert dialog_id in dialogs
    dialog = dialogs[dialog_id]

    return dialog

# Print entire conversation 
def show_dialog(dialog):
    for turn in dialog['turns']:
        if turn['speaker'] == 'USER':
            cprint('        User: ', 'blue', attrs=['bold'], force_color=True, end='')
            cprint(turn['utterance'], 'blue', force_color=True)
        else:
            cprint('AI Assistant: ', 'yellow', attrs=['bold'], force_color=True, end='')
            cprint(turn['utterance'], 'yellow', force_color=True)

# Print goals extracted from the data point
def show_dialog_goals(goals):
    for service_name, service_goals in goals.items():
        print('service: ', end='')
        cprint(service_name, 'red', attrs=['bold'], force_color=True)
        for intent_name, intent_goals in service_goals.items():
            print(f'  intent: ', end='')
            cprint(intent_name, 'green', attrs=['bold'], force_color=True)
            inform_str = [f"{s} = {vd['canonical_value']}" for s, vd in intent_goals['inform'].items()]
            inform_str = ', '.join(inform_str)
            print('inform:', inform_str)
            print('request:', intent_goals['request'])


# Extract user goals from the SGD data
def extract_user_goals(dialog):
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
                    goals[service_name][intent]['inform'][action['slot']] = {
                        'value': action['values'][0],
                        'canonical_value': action['canonical_values'][0],
                    }
                elif action['act'] == 'REQUEST':
                    goals[service_name][intent]['request'].append(action['slot'])

    goals = {k: v for k, v in goals.items() if v}
    for service_name, service in goals.items():
        service = {k: v for k, v in service.items() if v['inform'] or v['request']}
        for intent in service.values():
            intent['request'] = list(set(intent['request']))
        goals[service_name] = service

    return goals

# Turn user goals into string
def make_goals_str(goals):
    goals_str = []
    goal_index = 0
    for service_name, service in goals.items():
        intent_dict = {it['name']: it for it in schemas[service_name]['intents']}
        slot_dict = {slot['name']: slot for slot in schemas[service_name]['slots']}
        for intent_name, intent in service.items():
            intent_desc = intent_dict[intent_name]['description']
            intent_desc = intent_desc[0].lower() + intent_desc[1:]

            goal_index += 1
            goals_str.append(f'\nGoal {goal_index}:')

            goals_str.append(f'You want to {intent_desc}.')

            inform_str = []
            for slot, value_dict in intent['inform'].items():
                slot_desc = slot_dict[slot]['description']
                slot_desc = slot_desc[0].lower() + slot_desc[1:]
                slot_value_str = f"the {slot_desc} is {value_dict['value']}"
                if 'canonical_value' in value_dict and value_dict['canonical_value'] != value_dict['value']:
                    slot_value_str += f" ({value_dict['canonical_value']})"
                inform_str.append(slot_value_str)
            if inform_str:
                inform_str = ', '.join(inform_str) + '.'
                goals_str.append(f'You will inform the AI Assistant that: {inform_str}')

            request_str = []
            for slot in intent['request']:
                slot_desc = slot_dict[slot]['description']
                slot_desc = slot_desc[0].lower() + slot_desc[1:]
                request_str.append(f'the {slot_desc}')
            if request_str:
                request_str = ', '.join(request_str) + '.'
                goals_str.append(f'You ask the AI Assistant to know: {request_str}')

    goals_str = '\n'.join(goals_str).strip()
    return goals_str

# Combines extraction and conversion to string for the user goals
def prepare_goals_str(dialog):
    goals = extract_user_goals(dialog)
    goals_str = make_goals_str(goals)
    return goals_str

# Converts conversation into string
def make_dialog_str(dialog):
    role_map = {'USER': 'User', 'SYSTEM': 'AI Assistant'}
    dialog_str = [role_map[turn['speaker']] + ': ' + turn['utterance'] for turn in dialog['turns']]
    dialog_str = '\n'.join(dialog_str)
    return dialog_str