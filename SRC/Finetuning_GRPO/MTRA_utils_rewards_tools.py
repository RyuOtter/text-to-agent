from __future__ import annotations
import json
import re
import os
from collections import defaultdict
from datetime import datetime
from Modular_Learning_Agent.utils_llm import LLMModel

# Config for tool rewards
class ToolsRewardConfig:
    
    def __init__(self, config_dict):

        self.json_valid_reward = float(config_dict["json_valid_reward"])
        self.json_invalid_penalty = float(config_dict["json_invalid_penalty"])
        self.json_no_tool_reward = float(config_dict["json_no_tool_reward"])
        self.repetition_penalty = float(config_dict["repetition_penalty"])
        self.booking_no_reference_penalty = float(config_dict["booking_no_reference_penalty"])
        self.json_leakage_penalty = float(config_dict["json_leakage_penalty"])
        self.tools_reward_cap = float(config_dict["tools_reward_cap"])
        self.enable_json_validity = bool(config_dict["enable_json_validity"])
        self.enable_repetition_penalty = bool(config_dict["enable_repetition_penalty"])
        self.enable_booking_validation = bool(config_dict["enable_booking_validation"])
        self.enable_json_leakage_penalty = bool(config_dict["enable_json_leakage_penalty"])

# Penalty for tool repetition
def compute_repetition_penalty_from_tool_logs(tool_logs, config):
    if not config.enable_repetition_penalty or not tool_logs:
        return 0.0
    
    seen_calls = set()
    for tool_log in tool_logs:
        tool_name = tool_log.get("name", "")
        args = tool_log.get("args", {})
        signature = (tool_name.lower().strip(), json.dumps(args, sort_keys=True) if isinstance(args, dict) else str(args))
        if signature in seen_calls:
            return config.repetition_penalty
        else:
            seen_calls.add(signature)
    return 0.0 


# Detect invalid bookings
def compute_booking_validation_penalty_improved(sim_conv, tool_logs, eval_llm, config):
    if not config.enable_booking_validation:
        return 0.0
    
    conversation_text = ""
    for turn in sim_conv:
        user_msg = turn.get("user", "").strip()
        agent_msg = turn.get("agent", "").strip()
        
        if user_msg:
            conversation_text += f"USER: {user_msg}\n"
        if agent_msg:
            conversation_text += f"ASSISTANT: {agent_msg}\n"
    
    system_prompt = """You are an expert at analyzing booking conversations. Your task is to count how many bookings the assistant claims to have completed.

COUNT BOOKING CLAIMS when the assistant explicitly states that a booking/reservation has been COMPLETED:

CRITICAL: You MUST respond with ONLY a JSON object. No explanations, no text before or after.

EXPLICIT BOOKING CLAIMS (COUNT THESE):
- "I've booked your table at Pizza Palace"
- "Your reservation is confirmed"  
- "The booking has been made successfully"
- "I've reserved a table for you"
- "Your hotel room is booked"
- "The flight has been booked"

DO NOT COUNT:
- Search attempts: "I'm looking for restaurants"
- Booking intentions: "I'll book that for you" (future tense)
- Partial bookings: "I need more information to complete the booking"
- Failed attempts: "I tried to book but it failed"
- Questions: "Would you like me to book this restaurant?"

OUTPUT FORMAT: Return ONLY a JSON object:
{"booking_claims_count": <number>}
For example:
RESPOND WITH ONLY THIS JSON FORMAT:
{"booking_claims_count": 0}
OR
{"booking_claims_count": 1}
OR
{"booking_claims_count": 2}
"""

    user_prompt = f"Count booking completion claims in this conversation:\n\n{conversation_text}"
    
    try:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = eval_llm.chat(messages)
            if not response or not response.strip():
                return 0.0
            response_text = response.strip()
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as json_err:
                return 0.0
        except Exception as e:
            return 0.0
        
        booking_claims_count = result.get("booking_claims_count", 0)
        reference_count = 0
        for tool_log in tool_logs:
            tool_result = str(tool_log.get("result", ""))
            if "reference number" in tool_result.lower():
                reference_count += 1
        has_invalid_booking = booking_claims_count > reference_count
        penalty = config.booking_no_reference_penalty if has_invalid_booking else 0.0
        return penalty
    except Exception as e:
        return 0.0

# Detect JSON leakage
def compute_json_leakage_penalty_llm(sim_conv, eval_llm, config):
    if not config.enable_json_leakage_penalty or not eval_llm:
        return 0.0
    
    system_prompt = """You are an expert at detecting technical leakage in conversations.

CRITICAL: You MUST respond with ONLY a JSON object. No explanations, no text before or after.

DETECT JSON/TECHNICAL LEAKAGE when the assistant shows:
- Raw JSON: {"tool_name": "search_restaurants", "arguments": {"cuisine": "italian"}}
- Malformed JSON: {"tool_name": "search_restaurants" (missing closing bracket)
- Technical keywords: "tool_name", "function_name", "arguments", "parameters"
- Code-like syntax: function_call(), api.method(), <result>...</result>
- Internal system messages or debugging info
- Raw API responses or error codes

EXAMPLES OF NORMAL RESPONSES (NO PENALTY):
- "I found 3 Italian restaurants for you"
- "Let me search for restaurants in your area"
- "I'll book a table at Pizza Palace"
- "Here are the search results: Restaurant A, Restaurant B"
- Normal conversational responses with no technical exposure

IMPORTANT:
- Focus on what the USER would see in the final conversation
- Penalize any technical/internal information that leaked through
- Natural language descriptions of actions are fine
- Only penalize actual JSON, code, or technical formatting

RESPOND WITH ONLY THIS JSON
{"has_json_leakage": true}
OR
{"has_json_leakage": false}
NO OTHER TEXT ALLOWED."""

    user_prompt = f"Check for JSON/technical leakage in this conversation:\n\n{sim_conv}"
    
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = eval_llm.chat(messages)
        if not response or not response.strip():
            return 0.0
        response_text = response.strip()
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as json_err:
            return 0.0
        
        has_leakage = result.get("has_json_leakage", False)
        penalty = config.json_leakage_penalty if has_leakage else 0.0
        
        return penalty
    except Exception as e:
        return 0.0


# Episodic reward function for tools
class OutcomeToolsRewardFunction:
    
    def __init__(self, config, eval_llm=None):
        self.config = config
        self.eval_llm = eval_llm
        self.current_step = 0
        self.global_generation_idx = 0
        self.__name__ = "OutcomeToolsRewardFunction"

    
    # Compute episodic tool rewards for batch of completions
    def __call__(self, *, prompts, completions, raw_datapoint=None, **kwargs):

        rewards = []

        for i, sim_conv in enumerate(completions):
            datapoint = raw_datapoint[i] if raw_datapoint and i < len(raw_datapoint) else {}
            tool_logs_list = kwargs.get("tool_logs", [])
            assistant_only_list = kwargs.get("assistant_only", [])
            completion_tool_logs = tool_logs_list[i] if i < len(tool_logs_list) else []
            completion_assistant_only = assistant_only_list[i] if i < len(assistant_only_list) else ""
            episode_kwargs = dict(kwargs)
            episode_kwargs["tool_logs"] = completion_tool_logs
            episode_kwargs["assistant_only"] = completion_assistant_only

            total_reward, reward_breakdown = self._compute_single_episode_reward(sim_conv, datapoint, **episode_kwargs)
            capped_reward = min(max(total_reward, -self.config.tools_reward_cap), self.config.tools_reward_cap)
            rewards.append(capped_reward)

            if not hasattr(self, "_last_reward_breakdowns"):
                self._last_reward_breakdowns = []

            if len(self._last_reward_breakdowns) <= i:
                self._last_reward_breakdowns.extend([None] * (i + 1 - len(self._last_reward_breakdowns)))
            
            self._last_reward_breakdowns[i] = {"total_reward": total_reward, "capped_reward": capped_reward, "breakdown": reward_breakdown}
        
        self.current_step += 1

        return rewards
    
    # Compute tool reward for single conversation
    def _compute_single_episode_reward(self, sim_conv, datapoint, **kwargs):
        tool_logs = kwargs.get("tool_logs", [])
        assistant_only = kwargs.get("assistant_only", "")
        total_reward = 0.0
        reward_breakdown = {"components": {}, "cap": self.config.tools_reward_cap, "turns_count": len(sim_conv) if sim_conv else 0}
        real_tool_logs = tool_logs if tool_logs is not None else []

        json_reward = self._compute_json_validity_reward_from_tags(assistant_only, self.config)
        total_reward += json_reward
        reward_breakdown["components"]["json_validity"] = json_reward

        repetition_penalty = compute_repetition_penalty_from_tool_logs(real_tool_logs, self.config)
        total_reward += repetition_penalty
        reward_breakdown["components"]["repetition_penalty"] = repetition_penalty

        leakage_penalty = 0.0

        if self.eval_llm and sim_conv:
            leakage_penalty = compute_json_leakage_penalty_llm(sim_conv, self.eval_llm, self.config)
            total_reward += leakage_penalty
        reward_breakdown["components"]["json_leakage_penalty"] = leakage_penalty
        booking_penalty = 0.0

        if self.eval_llm and sim_conv:
            booking_penalty = compute_booking_validation_penalty_improved(sim_conv, real_tool_logs, self.eval_llm, self.config)
            total_reward += booking_penalty
        reward_breakdown["components"]["booking_validation_penalty"] = booking_penalty 

        return total_reward, reward_breakdown

    # Compute JSON validity rewards from tags
    def _compute_json_validity_reward_from_tags(self, assistant_only, config):
        if not config.enable_json_validity:
            return 0.0
        
        successful_patterns = [r"<A\d+_TOOL_CALL_VALID>", r"<A\d+_TOOL_CALL_CORRECTED>", r"<A\d+_TOOL_CALL_RETRY_SUCCESS>"]
        successful_count = sum(len(re.findall(pattern, assistant_only)) for pattern in successful_patterns)
        
        failed_patterns = [r"<A\d+_TOOL_CALL_MALFORMED>", r"<A\d+_TOOL_CALL_CORRECTION_FAILED>", r"<A\d+_TOOL_CALL_RETRY_FAILED>"]
        failed_count = sum(len(re.findall(pattern, assistant_only)) for pattern in failed_patterns)
        
        if successful_count == 0 and failed_count == 0:
            return config.json_no_tool_reward
        
        quality_good_count, quality_bad_count = self._analyze_successful_tool_quality(assistant_only)
        reward = (quality_good_count * config.json_valid_reward + quality_bad_count * config.json_invalid_penalty + failed_count * config.json_invalid_penalty)
        
        return reward
    
    # Analyze tool calls to identify if they have good or bad quality JSON
    def _analyze_successful_tool_quality(self, assistant_only):
        quality_good_count = 0
        quality_bad_count = 0
        
        successful_patterns = [r"<A\d+_TOOL_CALL_VALID>(.*?)</A\d+_TOOL_CALL_VALID>", r"<A\d+_TOOL_CALL_CORRECTED>(.*?)</A\d+_TOOL_CALL_CORRECTED>", r"<A\d+_TOOL_CALL_RETRY_SUCCESS>(.*?)</A\d+_TOOL_CALL_RETRY_SUCCESS>"]
        
        for pattern in successful_patterns:
            matches = re.findall(pattern, assistant_only, re.DOTALL)
            
            for match in matches:
                json_pattern = r"\{[^{}]*'tool'[^{}]*'args'[^{}]*\{[^{}]*\}[^{}]*\}"
                json_matches = re.findall(json_pattern, match, re.DOTALL | re.MULTILINE)
                
                if not json_matches:
                    simple_pattern = r"\{.*?'tool'.*?\}"
                    json_matches = re.findall(simple_pattern, match, re.DOTALL)
                
                for json_str in json_matches:
                    try:
                        tool_call = json.loads(json_str)
                        args = tool_call.get("args", {})
                        tool_name = tool_call.get("tool", "")
                        
                        has_quality_issues = False
                        
                        valid_args = self._get_valid_tool_arguments(tool_name)
                        
                        for arg_name, arg_value in args.items():
                            if valid_args and arg_name not in valid_args:
                                has_quality_issues = True
                                break
                                
                            if isinstance(arg_value, str):
                                if arg_value.strip() == "":
                                    has_quality_issues = True
                                    break
                                
                                invalid_values = ["unknown", "your current city", "inquire within", "n/a", "null", "none"]
                                if any(invalid in arg_value.lower() for invalid in invalid_values):
                                    has_quality_issues = True
                                    break
                        
                        if has_quality_issues:
                            quality_bad_count += 1
                        else:
                            quality_good_count += 1
                            
                    except (json.JSONDecodeError, KeyError):
                        quality_bad_count += 1
                        continue
        
        return quality_good_count, quality_bad_count
    
    # Not used
    def _get_valid_tool_arguments(self, tool_name):
        tool_schemas = {
            "Flights_3_SearchOnewayFlight": {
                "origin_city", "destination_city", "departure_date", 
                "airlines", "passengers", "flight_class", "number_checked_bags"
            },
            "Flights_3_SearchRoundtripFlights": {
                "origin_city", "destination_city", "departure_date", "return_date",
                "airlines", "passengers", "flight_class", "number_checked_bags"
            },
            "Flights_3_GetCityAirports": {
                "city"
            },
        }
        
        return tool_schemas.get(tool_name, set())

# Skeleton for outcome-based reward functions
def create_outcome_tools_reward_function(config_name="rew_cfg_1_tools", config_dict=None, eval_llm=None):
    
    config = ToolsRewardConfig(config_dict)
    reward_func = OutcomeToolsRewardFunction(config, eval_llm)
    reward_func.__name__ = f"tools_outcome_reward_{config_name}"
    return reward_func

# Wrapper to integrate reward functions
def make_turn_reward_func(config_name="rew_cfg_1_tools", config_dict=None, eval_llm=None):
    return create_outcome_tools_reward_function(config_name, config_dict, eval_llm)