import ast
import json
import os
import re
import yaml
import openai
from dotenv import load_dotenv
from termcolor import cprint
from .sgd_data_utils import make_dialog_str
from .sgd_tools import make_one_function_schema
from .utils_llm import LLMModel


# Create tool descriptions and instructions as string that can be added to the system prompt
def serialize_tools_for_prompt(tools, schemas=None):

    tool_strings = []

    for tool_name, tool in tools.items():
        lines = []
        lines.append(f"Tool name: {tool.name}")
        if schemas is not None:
            service_key = "_".join(tool.name.split("_")[:2])  
            intent_name = tool.name.split("_", maxsplit=2)[2]
            schema = schemas.get(service_key)
            intent = next((i for i in schema["intents"] if i["name"].lower() == intent_name.lower()), None)

            try:
                schema_info = make_one_function_schema(schema, intent_name)
            except Exception:
                schema_info = None

            func_desc = intent.get("description", "")
            kind = "Transaction function" if intent.get("is_transactional", False) else "Query function"

            required_args = intent.get("required_slots", [])
            optional_args = list(intent.get("optional_slots", {}).keys())
            all_args = required_args + optional_args
            arg_str = ", ".join(all_args) if all_args else "None"

            lines.append(f"Kind: {kind}")
            lines.append(f"Description: {func_desc}")
            lines.append(f"Arguments: {arg_str}")

            if schema_info and isinstance(schema_info, dict):
                params = schema_info.get("parameters", {})
                props = params.get("properties", {})
                req = set(params.get("required", []) or [])

                if props:
                    req_lines = []
                    for arg in [a for a in props.keys() if a in req]:
                        spec = props.get(arg, {})
                        ty = spec.get("type", "string")
                        extras = []
                        if "enum" in spec and spec["enum"]:
                            extras.append("enum: " + " | ".join(map(str, spec["enum"])))
                        if "examples" in spec and spec["examples"]:
                            ex = spec["examples"][:5]
                            extras.append("examples: " + ", ".join(map(str, ex)))
                        extra_str = f" ({'; '.join(extras)})" if extras else ""
                        req_lines.append(f"- {arg}: {ty}{extra_str}")
                    if req_lines:
                        lines.append("Required arguments:")
                        lines.extend(req_lines)

                    opt_lines = []
                    for arg in [a for a in props.keys() if a not in req]:
                        spec = props.get(arg, {})
                        ty = spec.get("type", "string")
                        extras = []
                        if "enum" in spec and spec["enum"]:
                            extras.append("enum: " + " | ".join(map(str, spec["enum"])))
                        if "examples" in spec and spec["examples"]:
                            ex = spec["examples"][:5]
                            extras.append("examples: " + ", ".join(map(str, ex)))
                        extra_str = f" ({'; '.join(extras)})" if extras else ""
                        opt_lines.append(f"- {arg}: {ty}{extra_str}")
                    if opt_lines:
                        lines.append("Optional arguments:")
                        lines.extend(opt_lines)

        else:
            lines.append(f"Description: {tool.description}")
            
            if hasattr(tool, "schema") and tool.schema:
                schema = tool.schema
                props = schema.get("parameters", {}).get("properties", {})
                required = set(schema.get("parameters", {}).get("required", []))
                
                if tool.required_args:
                    lines.append("Required arguments:")
                    for arg in tool.required_args:
                        if arg in props:
                            prop = props[arg]
                            arg_type = prop.get("type", "string")
                            arg_desc = prop.get("description", "")
                            extras = []
                            if "enum" in prop and prop["enum"]:
                                extras.append("enum: " + " | ".join(map(str, prop["enum"])))
                            extra_str = f" ({'; '.join(extras)})" if extras else ""
                            desc_str = f" - {arg_desc}" if arg_desc else ""
                            lines.append(f"- {arg}: {arg_type}{extra_str}{desc_str}")
                        else:
                            lines.append(f"- {arg}: (required)")
                
                if tool.optional_args:
                    lines.append("Optional arguments:")
                    for arg in tool.optional_args:
                        if arg in props:
                            prop = props[arg]
                            arg_type = prop.get("type", "string")
                            arg_desc = prop.get("description", "")
                            extras = []
                            if "enum" in prop and prop["enum"]:
                                extras.append("enum: " + " | ".join(map(str, prop["enum"])))
                            extra_str = f" ({'; '.join(extras)})" if extras else ""
                            desc_str = f" - {arg_desc}" if arg_desc else ""
                            lines.append(f"- {arg}: {arg_type}{extra_str}{desc_str}")
                        else:
                            lines.append(f"- {arg}: (optional)")
                
                if tool_name.startswith("query_"):
                    if "restaurant" in tool_name:
                        lines.append("\nValid SQL values:")
                        lines.append("- area: centre, north, south, east, west")
                        lines.append("- food: [any cuisine type - chinese, italian, british, etc.]")
                        lines.append("- pricerange: cheap, moderate, expensive")
                        lines.append("- name: [restaurant name]")
                        lines.append("- Note: `area = \"Cambridge\"` should be completely ignored")
                    elif "hotel" in tool_name:
                        lines.append("\nValid SQL values:")
                        lines.append("- area: centre, north, south, east, west")
                        lines.append("- pricerange: cheap, moderate, expensive")
                        lines.append("- type: hotel, guesthouse")
                        lines.append("- parking: yes, no")
                        lines.append("- internet: yes, no")
                        lines.append("- stars: 1, 2, 3, 4, 5, etc.")
                        lines.append("- name: [hotel name]")
                        lines.append("- Note: `area = \"Cambridge\"` should be completely ignored")
                    elif "attraction" in tool_name:
                        lines.append("\nValid SQL values:")
                        lines.append("- area: centre, north, south, east, west")
                        lines.append("- type: museum, college, sports, entertainment, gallery, pub, theatre, cinema, tourist, church, park, etc.")
                        lines.append("- name: [attraction name]")
                        lines.append("- Note: `area = \"Cambridge\"` should be completely ignored")
                    elif "train" in tool_name:
                        lines.append("\nValid SQL values:")
                        lines.append("- departure: [city name - cambridge, london, birmingham, etc.]")
                        lines.append("- destination: [city name - cambridge, london, birmingham, etc.]")
                        lines.append("- day: monday, tuesday, wednesday, thursday, friday, saturday, sunday")
                        lines.append("- leaveAt: [time format hh:mm, examples: 08:30, 16:00]")
                        lines.append("- arriveBy: [time format hh:mm, examples: 08:30, 16:00]")
                        lines.append("\nSQL Notes:")
                        lines.append("- When querying trains with leave time, it means querying trains leaving after that time, i.e. `WHERE \"leaveAt\" >= \"08:05\"`")
                        lines.append("- When querying trains with arrive time, it means querying trains arriving before that time, i.e. `WHERE \"arriveBy\" <= \"08:05\"`")
                        lines.append("- The departure and destination should be different")
                        lines.append("- The leave time should be earlier than the arrive time")
                    
            else:
                if tool.required_args:
                    lines.append("Required arguments:")
                    for arg in tool.required_args:
                        lines.append(f"- {arg}: (required)")
                
                if tool.optional_args:
                    lines.append("Optional arguments:")  
                    for arg in tool.optional_args:
                        lines.append(f"- {arg}: (optional)")
            
            if not tool.required_args and not tool.optional_args:
                lines.append("Arguments: None")

        tool_str = "\n".join(lines) + "\n"
        tool_strings.append(tool_str)

    final_str = "\n".join(tool_strings)
    return final_str


# Helper function to extract tool call from LLM response if JSON format is present
def extract_tool_json(text):

    s = text.strip()
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "tool" in obj and isinstance(obj.get("args"), dict):
            return obj
    except Exception:
        pass

    start = s.find("{")
    while start != -1:
        depth = 0
        for i, ch in enumerate(s[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = s[start:i + 1]
                    try:
                        obj = json.loads(cand)
                        if isinstance(obj, dict) and "tool" in obj and isinstance(obj.get("args"), dict):
                            return obj
                    except Exception:
                        break 
        start = s.find("{", start + 1)

    return None


# Second fallback if extract_tool_json fails as JSON is not perfect but malformed
def looks_like_malformed_tool_call(text):
    
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    
    has_curly_braces = "{" in text and "}" in text
    has_tool_keyword = re.search(r"'tool'\s*:", text, re.IGNORECASE) is not None
    has_args_keyword = re.search(r"'args'\s*:", text, re.IGNORECASE) is not None
    has_json_structure = re.search(r"{\s*'[^']+'\s*:\s*", text) is not None
    
    return has_curly_braces and (has_tool_keyword or (has_args_keyword and has_json_structure))



# Agent class used to initialize user and assistant agents with tool calling logic
class Agent:

    # Initialize agents
    def __init__(self, name, system_prompt, model, role="assistant", tools=None):
        self.name = name
        self.prompt_template = system_prompt
        self.model = model
        self.role = role
        self.goal = None
        self.tools = tools or {}
        self.tool_log = []
        self.schemas = None
        self.grpo_training = False
        self.action_trace = []
        self.prompt = system_prompt

    # Set the tools for the agent and update system prompt
    def set_tools(self, tools, schemas=None):

        self.tools = tools or {}
        if schemas is not None:
            self.schemas = schemas

        if self.role == "assistant":
            tool_list_str = serialize_tools_for_prompt(self.tools, self.schemas)
            self.prompt = self.prompt_template.replace("{tool_list}", tool_list_str)

    # Reset the action trace for GRPO training
    def _reset_action_trace(self):
        self.action_trace = []

    # Classifies assistant actions for reward computation in GRPO training
    def _classify_assistant_action(self, response, context=""):
        tool_call = extract_tool_json(response)
        if tool_call is not None:
            if context == "schema_correction":
                return "TOOL_CALL_CORRECTED"
            elif context == "retry_after_failure":
                return "TOOL_CALL_RETRY_SUCCESS"
            elif context == "single_retry":
                return "TOOL_CALL_RETRY_SUCCESS"
            elif context == "general_correction":
                return "TOOL_CALL_CORRECTED"
            else:
                return "TOOL_CALL_VALID"
        
        if looks_like_malformed_tool_call(response):
            if context == "schema_correction":
                return "TOOL_CALL_CORRECTION_FAILED"
            elif context == "retry_after_failure":
                return "TOOL_CALL_RETRY_FAILED"
            elif context == "single_retry":
                return "TOOL_CALL_RETRY_FAILED"
            elif context == "general_correction":
                return "TOOL_CALL_CORRECTION_FAILED"
            else:
                return "TOOL_CALL_MALFORMED"
        
        return "REPLY"

    # Add user goal into the user assistants system prompt
    def include_user_goal(self, task_goal):
        self.goal = task_goal
        self.prompt = self.prompt_template.replace("{user_goals}", self.goal)


    # Main call method for agents to generate a response
    def __call__(self, simulated_conversation, ground_truth_example = None, rl_recorder = None, policy_seed = None):

        conversation_history = ""
        for turn in simulated_conversation:
            if "user" in turn:
                conversation_history += f"user: {turn['user']}\n"
            if "agent" in turn:
                conversation_history += f"assistant: {turn['agent']}\n"
        
        filled_prompt = self.prompt.replace("{conversation}", conversation_history.strip())

        if self.role == "user" and ground_truth_example is not None:
            filled_prompt = filled_prompt.replace("{reference_dialogue}", make_dialog_str(ground_truth_example))
        elif self.role == "user" and "{reference_dialogue}" in filled_prompt:
            filled_prompt = filled_prompt.replace("{reference_dialogue}", "")
        
        message = [{"role": "system", "content": filled_prompt}]

        if self.role == "user":
            user_out = self.model["dialogue"].chat(message)
            return user_out

        response = self.model["dialogue"].chat(message)
        
        if self.role == "assistant" and self.grpo_training:
            kind = self._classify_assistant_action(response, "initial")
            self.action_trace.append((kind, response))

        last_user_text = None
        for turn in reversed(simulated_conversation):
            if "user" in turn and turn["user"]:
                last_user_text = turn["user"]
                break
        
        # Starts tool call detection logic
        if self.role == "assistant" and self.tools:
            response = self._maybe_handle_tool_call(message, response, conversation_history, last_user=last_user_text)
            return response
                
        return response
    
    
    # Tool call detection logic
    def _maybe_handle_tool_call(self, messages, response, conversation_history, retrying=False, retries=0, max_retries=3, last_user = None):

        tool_call = extract_tool_json(response)
        if tool_call is not None and isinstance(tool_call, dict):
            raw_name = str(tool_call.get("tool", "")).strip()
            args = tool_call.get("args", {}) or {}
            if not isinstance(args, dict):
                args = {}

            tool_key = raw_name
            if tool_key not in self.tools:
                lower_map = {k.lower(): k for k in self.tools.keys()}
                tool_key = lower_map.get(raw_name.lower(), raw_name)

            if tool_key in self.tools and args is not None:
                return self._execute_tool(
                    tool_key, args, messages, response, conversation_history,
                    retrying=retrying, retries=retries, max_retries=max_retries, last_user=last_user
                )

        clarification_msg = "You were supposed to use a tool by responding with a strict JSON object like this:\n{\"tool\": \"tool_name\", \"args\": {\"param1\": \"value1\", \"param2\": \"value2\"}}\n\nHowever, your response was not valid JSON or had missing keys.\nPlease ask the user for any missing information, or retry using the correct format."
        retry_history = messages + [
            {"role": "assistant", "content": response},
            {"role": "system", "content": clarification_msg},
        ]

        final_response = self.model["correction"].chat(retry_history)
        if self.grpo_training:
            kind = self._classify_assistant_action(final_response, "general_correction")
            self.action_trace.append((kind, final_response))

        if looks_like_malformed_tool_call(final_response):
            final_response = "I'd be happy to help you with that. Could you provide a bit more detail about what you're looking for?"

        return final_response


    # Executes of tool call if correct JSON format has been detected
    def _execute_tool(self, tool_name, args, messages, response, conversation_history, retrying=False, retries=0, max_retries=3, last_user = None):

        tool = self.tools.get(tool_name)
        try:
            tool_result = tool(**args)
            if tool_result is None:
                tool_result = "No results found."
            elif not isinstance(tool_result, str):
                tool_result = str(tool_result)
            
        except Exception as e:
            expected_args = getattr(tool, "arg_names", [])

            if retrying and retries < max_retries:
                failed_calls = [
                    log for log in self.tool_log
                    if isinstance(log["result"], str) and ("no result" in log["result"].lower() or "empty" in log["result"].lower())
                ]
                tool_log_str = "\n".join([
                    f"- {log['name']} | args: {log['args']} | result: {str(log['result'])[:100]}"
                    for log in failed_calls
                ]) or "None"

                tool_list_str = serialize_tools_for_prompt(self.tools, self.schemas)
                reflective_prompt = [
                    {"role": "system", "content": f"Conversation so far:\n\n{conversation_history.strip()}"},
                    {"role": "system", "content": f"Your last tool call failed again.\nHere are all previous tool calls attempted:\n\n{tool_log_str}\n\nReflect on possible issues:\n- Are you using the correct tool?\n- Could any arguments be phrased differently (e.g. 'Burgers' → 'American')?\n- Did you provide too many or redundant arguments?\n- Are you missing important ones?\n- Please use canonical full formats and YYYY-MM-DD where needed.\n\nBased on the full context, please try calling a tool again.\nRespond using valid JSON: {{\"tool\": ..., \"args\": {{...}}}}.\n"},
                    {"role": "system", "content": f"Here is the full list of available tools:\n\n{tool_list_str}"}
                ]

                retry_response = self.model["correction"].chat(reflective_prompt)
                if self.grpo_training:
                    kind = self._classify_assistant_action(retry_response, "retry_after_failure")
                    self.action_trace.append((kind, retry_response))

                return self._maybe_handle_tool_call(
                    messages, retry_response, conversation_history,
                    retrying=True, retries=retries + 1, max_retries=max_retries, last_user=last_user
                )

            retry_instruction = "The previous function call failed. Please rewrite the tool call using canonical values and respond with a valid JSON tool call only."
            retry_prompt = [
                {"role": "system", "content": "Your last reply wasn't a valid JSON tool call. Respond with ONLY a JSON object: {\"tool\": \"<name>\", \"args\": {…}}. Do not include explanations or markdown."},
            ]
            if last_user is not None:
                retry_prompt.append({"role": "user", "content": last_user})
            retry_prompt.extend([
                {"role": "assistant", "content": response},
                {"role": "system", "content": retry_instruction},
            ])
            retry_response = self.model["dialogue"].chat(retry_prompt)
            if self.grpo_training:
                kind = self._classify_assistant_action(retry_response, "single_retry")
                self.action_trace.append((kind, retry_response))

            return self._maybe_handle_tool_call(
                messages, retry_response, conversation_history,
                retrying=True, retries=retries + 1, max_retries=max_retries, last_user=last_user
            )

        self.tool_log.append({"name": tool_name, "args": args, "result": tool_result})

        followup_history = [
            {
                "role": "system",
                "content": "Using the tool result below, write a concise, friendly assistant reply to the user.\nRequirements:\n- Include the exact values for any attributes the user requested that appear in the result.\n- Use canonical formats: dates YYYY-MM-DD, times HH:MM, prices as digits only (no $), booleans 'True'/'False'.\n- Do not paraphrase slot values; state them verbatim where possible.\n- If multiple rows were returned, either ask one clarifying question to choose, or pick the first row deterministically and proceed.\n- Do NOT paste raw tables or mention tools.\n- While you MUST use the exact value returned by the tool call for a given attribute, make your answer natural and human-like."
            },
        ]
        if last_user:
            followup_history.append({"role": "user", "content": last_user})

        followup_history.append({"role": "function", "name": tool_name, "content": tool_result})

        assistant_followup = self.model["dialogue"].chat(followup_history)
        if self.grpo_training:
            self.action_trace.append(("REPLY", assistant_followup))

        return assistant_followup