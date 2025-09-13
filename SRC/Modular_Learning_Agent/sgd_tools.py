# Disclaimer: This file consists entirely of code from the AutoTOD codebase
###########################################################################

import random
import sqlite3
import os
from .sgd_data_utils import INFO_DB_PATH, TRANS_DB_PATH, load_schemas, load_dialogs
from collections import defaultdict

# Configuration
schemas = None

# Ensure that the service with its parameters is valid
def sgd_function_check(service_name, intent_name, args):
    global schemas
    if schemas is None:
        schemas = load_schemas()
    if service_name not in schemas:
        return False, f'Service "{service_name}" does not exist.'
    schema = schemas[service_name]

    intent_dict = {intent['name']: intent for intent in schema['intents']}
    if intent_name not in intent_dict:
        return False, f'Service "{service_name}" does not have the intent "{intent_name}".'
    intent = intent_dict[intent_name]
    
    if missing_args := [arg for arg in intent['required_slots'] if arg not in args]:
        args_str = ', '.join(f'"{x}"' for x in missing_args)
        return False,  f'The required parameters {args_str} are missing.'
    
    if error_args := [arg for arg in args if arg not in intent['required_slots'] + list(intent['optional_slots'].keys())]:
        error_args_str = ', '.join(f'"{x}"' for x in error_args)
        required_args_str = ', '.join(f'"{x}"' for x in intent['required_slots'])
        optional_args_str = ', '.join(f'"{x}"' for x in intent['optional_slots'])
        msg = f'Parameters {error_args_str} are not valid. Please provide valid parameters.'
        msg += f'Required parameters {required_args_str} and optional parameters {optional_args_str}.'
        return False, msg
    
    return True, 'ok'

# Create tables for SQL queries that are easier to comprehend for the LLM
def make_table_string(cursor, max_items=5, max_chars=500):
    records = cursor.fetchall()

    if len(records) == 0:
        return 'No results found.'

    result = []
    n_chars = 0

    line = '| ' + ' | '.join(desc[0] for desc in cursor.description) + ' |'
    n_chars += len(line) + 1
    result.append(line)

    line = '| ' + ' | '.join(['---'] * len(cursor.description)) + ' |'
    n_chars += len(line) + 1
    result.append(line)

    for i, record in enumerate(records, start=1):
        line = '| ' + ' | '.join(str(v) for v in record) + ' |'
        n_chars += len(line) + 1
        if i > 0 and n_chars <= max_chars and i <= max_items:
            result.append(line)
        else:
            n_left = len(records) - i + 1
            result.append(f'\n{n_left} more records ...')
            break

    result = '\n'.join(result)
    return result

# Execute SGD SQL query and return table results
def sgd_function_info(service_name, intent, args, db_path=INFO_DB_PATH):
    fields = ', '.join(f'"{field}"' for field in intent['result_slots'])
    sql = f'SELECT {fields} FROM {service_name}'
    if args:
        conditions = ' AND '.join(f'"{k}" = "{v}"' for k, v in args.items())
        sql += f' WHERE {conditions}'

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(sql)
    except Exception as e:
        return f'SQL failed: {e.__class__.__name__}: {e}'

    return make_table_string(cursor)

# Execute SGD SQL transaction tool and return reference number
def sgd_function_trans(service_name, args, db_path=TRANS_DB_PATH):

    # Generate random reference number
    def generate_reference_num():
        return ''.join(random.choice('abcdefghijklmnopqrstuvwxyz0123456789') for i in range(8))
    
    refer_number = generate_reference_num()
    args['refer_number'] = refer_number

    fields = ', '.join(f'"{field}"' for field in args.keys())
    value_syms = ', '.join(['?'] * len(args))
    sql = f'INSERT INTO {service_name}_Transaction ({fields}) VALUES ({value_syms})'

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(sql, list(args.values()))
    except Exception as e:
        return f'SQL failed: {e.__class__.__name__}: {e}'
    conn.commit()
    cursor.close()
    conn.close()

    return f'Transaction succeed. The reference number is {refer_number}.'

# Main SGD function that validates and executes queries or transactions
def sgd_function(service_name, intent_name, info_db_path=INFO_DB_PATH, trans_db_path=TRANS_DB_PATH, **kwargs):
    
    passed, msg = sgd_function_check(service_name, intent_name, kwargs)
    if not passed:
        return msg

    schema = schemas[service_name]

    intent_dict = {intent['name']: intent for intent in schema['intents']}
    intent = intent_dict[intent_name]

    if not intent['is_transactional']:
        return sgd_function_info(service_name, intent, kwargs, info_db_path)
    else:
        return sgd_function_trans(service_name, kwargs, trans_db_path)


# Collect database records from service results
def collect_db_records(dialogs):
    tables = defaultdict(list)

    for dialog in dialogs.values():
        for turn in dialog['turns']:
            if turn['speaker'] != 'SYSTEM':
                continue
            for frame in turn['frames']:
                if 'service_results' not in frame:
                    continue
                for result in frame['service_results']:
                    tables[frame['service']].append(result)

    for name, table in tables.items():
        d = {str(a): a for a in table}
        tables[name] = list(d.values())

    return tables


# Detect data types for database fields
def detect_field_data_type(tables):
    global schemas
    if schemas is None:
        schemas = load_schemas()

    # Check if string can be converted to float
    def is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    field_data_type = defaultdict(dict)
    for service_name, schema in schemas.items():
        table = tables[service_name]
        for slot in schema['slots']:
            field = slot['name']
            if all(item[field].isdigit() for item in table if field in item):
                field_data_type[service_name][field] = 'integer'
            elif all(is_float(item[field]) for item in table if field in item):
                field_data_type[service_name][field] = 'number'
            elif all(item[field].lower() in ['true', 'false'] for item in table if field in item):
                field_data_type[service_name][field] = 'boolean'
            else:
                field_data_type[service_name][field] = 'string'

    return field_data_type


# Load dialogs and detect data types
def get_field_data_type():
    dialogs = load_dialogs()
    tables = collect_db_records(dialogs)
    field_data_type = detect_field_data_type(tables)
    return field_data_type


# Configuration
field_data_type = None


# Create function schema for single SGD service intent
def make_one_function_schema(service_schema, intent_name):
    global field_data_type
    if field_data_type is None:
        field_data_type = get_field_data_type()
    
    service_name = service_schema['service_name']
    intent_dict = {intent['name']: intent for intent in service_schema['intents']}
    intent = intent_dict[intent_name]

    func_schema = {
        'name': f'{service_name}_{intent_name}',
        'description': None,
        'parameters': {
            'type': 'object',
            'properties': {},
            'required': intent['required_slots'].copy(),
        }
    }

    desc = intent['description'] + '.'
    if not intent['is_transactional']:
        desc += ' (Query function. Return db recored that meets conditions.)'
    else:
        desc += ' (Transaction function. Return a reference number when calling succeeds.)'
    func_schema['description'] = desc

    slot_dict = {slot['name']: slot for slot in service_schema['slots']}
    for slot_name in intent['required_slots'] + list(intent['optional_slots'].keys()):
        slot = slot_dict[slot_name]

        service_name = service_schema['service_name']
        if slot_name in field_data_type[service_name]:
            field_type = field_data_type[service_name][slot_name]
        elif set(slot['possible_values']) == {'True', 'False'}:
            field_type = 'boolean'
        else:
            field_type = 'string'

        type_func = {'string': str, 'integer': int, 'number': float, 'boolean': lambda s: s.lower() == 'true'}
        if slot['possible_values']:
            possible_values = list(map(type_func[field_type], slot['possible_values']))
            
        property_schema = {
            'type': field_type,
            'description': slot['description'],
        }

        if slot['possible_values'] and field_type != 'boolean':
            if slot['is_categorical']:
                property_schema['enum'] = possible_values
            else:
                property_schema['examples'] = possible_values

        func_schema['parameters']['properties'][slot_name] = property_schema

    return func_schema


# Create function schemas for all intents in given services
def make_function_schemas(service_name_list):
    global schemas
    if schemas is None:
        schemas = load_schemas()
    
    functions = []
    for service_name in service_name_list:
        service_schema = schemas[service_name]
        for intent in service_schema['intents']:
            func_schema = make_one_function_schema(service_schema, intent['name'])
            functions.append(func_schema)
    return functions

# Create system prompt for SGD function schema
def make_system_prompt(self):

    services_info = []
    for service_name in self.service_names:
        service_schema = self.sgd_schemas[service_name]
        functions_info = []
        for intent in service_schema['intents']:
            func_info = f'- {service_name}_{intent["name"]}: {intent["description"]}.'
            if not intent['is_transactional']:
                func_info += ' (Query function)'
            else:
                func_info += ' (Transaction function)'
            functions_info.append(func_info)
        functions_info = '\n'.join(functions_info)
        service_info = SERVICE_TEMPLATE.format(service_name=service_name, 
                                            service_desc=service_schema['description'],
                                            functions_info=functions_info)
        services_info.append(service_info)
    services_info = '\n\n'.join(services_info)