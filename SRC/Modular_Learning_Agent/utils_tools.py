from functools import partial
from .sgd_tools import sgd_function, make_one_function_schema
from .multiwoz_tools import prepare_query_db_functions, prepare_book_functions

# Build tools for the benchmark
def get_tools(benchmark, datapoint=None, schemas=None, dynamic=False):
    benchmark = benchmark.lower()
    if benchmark == "sgd":
        if dynamic:
            return get_tools_sgd(individual_dialog=datapoint, schemas=schemas)
        else:
            return {}
    elif benchmark == "multiwoz":
        return get_tools_multiwoz(datapoint=datapoint)
    else:
        raise NotImplementedError("Benchmark not implemented")

# Build tools for SGD
def get_tools_sgd(individual_dialog=None, schemas=None):
    tools = {}

    if schemas is None:
        raise ValueError("Schemas needed for SGD tool construction")

    if individual_dialog is not None:
        services = {frame["service"] for turn in individual_dialog["turns"] for frame in turn["frames"]}
    else:
        services = list(schemas.keys())

    for service_name in services:
        if service_name not in schemas:
            continue
        
        service_schema = schemas[service_name]

        for intent in service_schema["intents"]:
            intent_name = intent["name"]
            tool_name = f"{service_name}_{intent_name}"

            schema_info = make_one_function_schema(service_schema, intent_name)
            tool_description = schema_info["description"]
            required_args = schema_info.get("parameters", {}).get("required", [])
            all_args = list(schema_info.get("parameters", {}).get("properties", {}).keys())
            optional_args = [arg for arg in all_args if arg not in required_args]

            def tool_func_factory(s, i):
                return lambda **kwargs: sgd_function(s, i, **kwargs)

            tools[tool_name.lower()] = Tool(
                name=tool_name,
                description=tool_description,
                func=tool_func_factory(service_name, intent_name),
                required_args=required_args,
                optional_args=optional_args
            )

    return tools

# Build tools for MultiWOZ
def get_tools_multiwoz(datapoint=None):
    
    tools = {}
    query_domains = ["restaurant", "hotel", "attraction", "train"]

    for domain in query_domains:
        result = prepare_query_db_functions(domain)
        tool_name = result["name"]
        tool_func = result["function"] 
        schema = result["schema"]
        
        required_args = schema["parameters"].get("required", [])
        all_args = list(schema["parameters"].get("properties", {}).keys())
        optional_args = [arg for arg in all_args if arg not in required_args]
        
        tool = Tool(
            name=tool_name,
            description=schema["description"],
            func=tool_func,
            required_args=required_args,
            optional_args=optional_args
        )
        tool.schema = schema
        tools[tool_name] = tool
    
    booking_domains = ["restaurant", "hotel", "train", "taxi"]
    for domain in booking_domains:
        result = prepare_book_functions(domain)
        tool_name = result["name"]
        tool_func = result["function"]
        schema = result["schema"]
        
        required_args = schema["parameters"].get("required", [])
        all_args = list(schema["parameters"].get("properties", {}).keys())
        optional_args = [arg for arg in all_args if arg not in required_args]
        
        tool = Tool(
            name=tool_name,
            description=schema["description"],
            func=tool_func,
            required_args=required_args,
            optional_args=optional_args
        )
        tool.schema = schema
        tools[tool_name] = tool
    
    return tools

# Tool class used to make tools callable
class Tool:

    # Initialization of the tool
    def __init__(self, name, description, func, required_args=None, optional_args=None):
        self.name = name
        self.description = description
        self.func = func
        self.required_args = required_args or []
        self.optional_args = optional_args or []

    # Get the arguments of the tool
    @property
    def arg_names(self):
        return self.required_args + self.optional_args

    # Execute the tool
    def __call__(self, **kwargs):
        return self.func(**kwargs)