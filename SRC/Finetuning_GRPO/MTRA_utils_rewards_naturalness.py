from __future__ import annotations
import gc
import json
import re
import os
from collections import defaultdict
from datetime import datetime
from Modular_Learning_Agent.utils_llm import LLMModel

# Class for turn-level naturalness rewards
class TurnRewardConfigNaturalness:
    
    def __init__(self, config_dict):
        self.cost_per_turn = float(config_dict["cost_per_turn"])
        self.overlong_penalty = float(config_dict["overlong_penalty"])
        self.overlong_threshold = int(config_dict["overlong_threshold"])
        
        self.clarification_reward = float(config_dict["clarification_reward"])
        self.task_progression_min = float(config_dict["task_progression_min"])
        self.task_progression_max = float(config_dict["task_progression_max"])
        
        self.naturalness_judge_min = float(config_dict["naturalness_judge_min"])
        self.naturalness_judge_max = float(config_dict["naturalness_judge_max"])
        
        self.conciseness_cap_negative = float(config_dict["conciseness_cap_negative"])
        self.naturalness_cap_positive = float(config_dict["naturalness_cap_positive"])
        self.naturalness_cap_negative = float(config_dict["naturalness_cap_negative"])
        
        self.enable_conciseness = bool(config_dict["enable_conciseness"])
        self.enable_helpfulness = bool(config_dict["enable_helpfulness"])
        self.enable_naturalness = bool(config_dict["enable_naturalness"])


# Count tokens via tokenizer
def count_tokens_precise(text, tokenizer):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


# Conciseness reward function
def compute_conciseness_rewards(assistant_messages, config, tokenizer):

    if not config.enable_conciseness:
        return 0.0, {}
    
    total_reward = 0.0
    breakdown = {"cost_per_turn_penalty": 0.0, "overlong_penalty": 0.0, "turn_details": []}

    # Cost per turn
    cost_penalty = len(assistant_messages) * config.cost_per_turn
    total_reward += cost_penalty
    breakdown["cost_per_turn_penalty"] = cost_penalty
    
    # Cost for too long
    overlong_count = 0
    for i, message in enumerate(assistant_messages):
        token_count = count_tokens_precise(message, tokenizer)
        is_overlong = token_count > config.overlong_threshold
        
        turn_detail = {
            "turn_idx": i,
            "token_count": token_count,
            "is_overlong": is_overlong,
            "penalty": config.overlong_penalty if is_overlong else 0.0
        }
        breakdown["turn_details"].append(turn_detail)
        
        if is_overlong:
            total_reward += config.overlong_penalty
            overlong_count += 1
    
    breakdown["overlong_penalty"] = overlong_count * config.overlong_penalty
    breakdown["overlong_count"] = overlong_count
    
    return total_reward, breakdown


# Helpfulness reward function
def compute_helpfulness_rewards(completion, eval_llm, config):

    if not config.enable_helpfulness:
        return 0.0, {}
    
    total_reward = 0.0
    breakdown = {"clarification_reward": 0.0, "task_progression_reward": 0.0, "task_progression_raw_score": 0}
    
    try:
        has_clarification = evaluate_clarification_questions_llm(completion, eval_llm)
        clarification_reward = config.clarification_reward if has_clarification else 0.0
        total_reward += clarification_reward
        breakdown["clarification_reward"] = clarification_reward
        
        task_progress_score = evaluate_task_progression_llm(completion, eval_llm)
        task_reward = map_score_to_range(
            task_progress_score, 1, 5,
            config.task_progression_min, config.task_progression_max
        )
        total_reward += task_reward
        breakdown["task_progression_reward"] = task_reward
        breakdown["task_progression_raw_score"] = task_progress_score
        
    except Exception as e:
        print("Helpfulness reward failed")
    
    return total_reward, breakdown


# Identificiation of clairification questions
def evaluate_clarification_questions_llm(completion, eval_llm):

    system_prompt = """You are an expert at identifying clarification questions in conversations.

Your task is to determine whether the assistant asks any clarification questions to better understand the user's needs.

CLARIFICATION QUESTIONS include:
- Asking for missing information: "What time would you prefer?", "What is your starting city?", "When would you like to travel?"
- Getting preferences: "Would you like Italian or Chinese food?", "Do you prefer economy or business class?"
- Confirming understanding: "Do you mean downtown Seattle?", "You want a one-way flight, correct?"
- Requesting details: "How many people will be dining?", "How many passengers?", "What date did you have in mind?"
- Location details: "Where are you departing from?", "What's your destination city?", "Which airport?"
- Travel details: "What's your departure date?", "Any airline preference?", "What time works best?"

NOT CLARIFICATION QUESTIONS:
- Rhetorical questions: "Isn't that great?", "Don't you think so?"
- Questions about tool results: "Did the search work?", "Are these results helpful?"
- General conversation questions not seeking specific information: "How are you today?", "Are you excited about your trip?"
- System-revealing statements that expose internal workings:
  - "I'm having trouble with the tool call format"
  - "Let me clarify the system's capabilities"
  - "I'll send a message to the user to ask..."
  - Any mention of "tool", "system", "format", "capabilities" in error contexts
- Technical apologies or error explanations that reveal the assistant is using tools/systems

EVALUATION RULES:
- Return 1 if the assistant asks any clarification questions during the conversation
- Return 0 if no clarification questions are asked
- Focus on whether clarification happens, not how many times
- Ignore system-revealing or technical error messages

OUTPUT FORMAT:
Return only a JSON object following this template:
{"has_clarification": 0 or 1}"""

    assistant_messages = [msg["content"] for msg in completion if msg["role"] == "assistant"]
    assistant_text = "ASSISTANT MESSAGES:\n" + "\n".join([f"- {msg}" for msg in assistant_messages])
    
    user_prompt = f"{assistant_text}\n\nDoes the assistant ask any clarification questions?"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = eval_llm.chat(messages)
        response_lines = response.strip().split("\n")
        json_line = response_lines[0].strip()
        result = json.loads(json_line)
        return int(result.get("has_clarification", 0))
    except Exception as e:
        return 0

# Identify task progression
def evaluate_task_progression_llm(completion, eval_llm):

    system_prompt = """You are an expert at evaluating task progression and helpfulness in conversations.

Your task is to rate how effectively the assistant progresses toward completing the user's original goal.

IMPORTANT:
Use the full 1-5 scale. Most conversations should score 2-4 with only exceptional efficiency getting 5 and only complete failures getting 1.

EVALUATION CRITERIA:

Score 5 - EXCELLENT TASK PROGRESSION:
- Immediately understands user intent and acts efficiently
- Completes the task successfully with minimal back-and-forth (4-6 turns)
- Asks smart clarifying questions that advance the goal
- Maintains focus on user's original intent throughout
- Proactive and anticipates next steps

Score 4 - GOOD TASK PROGRESSION:
- Understands intent quickly and makes steady progress
- Completes most of the task with minor inefficiencies (6-8 turns)
- Asks appropriate clarifying questions when needed
- Stays mostly focused on the right entities
- Makes meaningful progress most turns

Score 3 - MODERATE TASK PROGRESSION:
- Takes time to understand intent but eventually gets there
- Makes some progress with noticeable inefficiencies (8+ turns)
- Gets sidetracked occasionally but recovers
- Reactive approach, needs user guidance
- Mixed success but shows effort toward the goal

Score 2 - POOR TASK PROGRESSION:
- Struggles to understand what user actually wants
- Gets stuck in repetitive patterns without real progress
- Confused about entities or loses track of the goal
- Many turns with little advancement toward completion
- Fails to complete the main task despite attempts

Score 1 - NO TASK PROGRESSION:
- Completely fails to understand or work toward user's goal
- Makes no meaningful progress in any direction
- System exposure or technical errors dominate
- Focuses on wrong entities or creates confusion
- Actively hinders task completion

FOCUS ON:
- Understanding of the user's original intent and goal
- Meaningful progress toward task completion
- Staying focused on relevant entities (e.g., correct cities, movies, etc.)
- Avoiding repetitive loops that don't advance the conversation
- Proactive vs reactive helpfulness
- Efficiency in gathering necessary information

EXAMPLES:

EXCELLENT (Score 5): User wants flight to San Diego > Assistant asks origin city > gets date > searches flights > finds good options > completes in 6 turns
GOOD (Score 4): User wants movie info > Assistant asks for preferences > finds movie > provides details > minor clarification needed > completes in 8 turns  
MODERATE (Score 3): User wants flight > Takes several turns to understand > gets confused about dates > eventually searches correctly > completes in 10 turns
POOR (Score 2): User wants flight > Repeats same questions > gets stuck on wrong city > makes failed searches > little real progress after 12+ turns
NO PROGRESS (Score 1): User wants flight > Assistant exposes system errors > focuses on wrong task > no meaningful progress toward booking

OUTPUT FORMAT: Return only a JSON object:
{"score": 1-5}"""

    dialogue_text = "CONVERSATION:\n"
    user_goal = ""
    
    for i, msg in enumerate(completion):
        if msg["role"] == "user" and i == 0:
            
            user_goal = msg["content"]
        
        if msg["role"] in ["user", "assistant"]:
            content = msg["content"]
            if msg["role"] == "user" and content.startswith("<result>"):
                dialogue_text += f"SYSTEM: [Tool result available]\n"
            else:
                dialogue_text += f"{msg['role'].upper()}: {content}\n"
    
    user_prompt = f"{dialogue_text}\n User's original goal: {user_goal}\n Evaluate how well the assistant progresses toward completing this goal:"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = eval_llm.chat(messages)
        response_lines = response.strip().split("\n")
        json_line = response_lines[0].strip()
        
        result = json.loads(json_line)
        score = int(result.get("score", 3))
        return score
    except Exception as e:
        return 3

# Compute naturalness rewards
def compute_naturalness_rewards(completion, eval_llm, config):
    
    if not config.enable_naturalness:
        return 0.0, {}
    
    breakdown = {"naturalness_score": 0.0, "naturalness_raw_score": 0}
    
    try:
        naturalness_score = evaluate_naturalness_llm(completion, eval_llm, config)
        naturalness_reward = map_score_to_range(
            naturalness_score, 1, 5,
            config.naturalness_judge_min, config.naturalness_judge_max
        )
        breakdown["naturalness_score"] = naturalness_reward
        breakdown["naturalness_raw_score"] = naturalness_score
        
        return naturalness_reward, breakdown
        
    except Exception as e:
        return 0.0, breakdown

# Assess naturalness of conversation
def evaluate_naturalness_llm(completion, eval_llm, config):
    
    system_prompt = """You are an expert evaluator of conversational naturalness and human-likeness.

Your task is to rate how natural, human-like, and conversational the assistant's responses sound, focusing on language style and phrasing rather than task completion.

IMPORTANT:
Use the full 1-5 scale. Most conversations should score 2-4, with only exceptional quality getting 5 and only major system violations getting 1.

EVALUATION CRITERIA - FOCUS ON LANGUAGE & PHRASING:
- Natural speech patterns: Uses contractions, varied sentence structure, natural transitions
- Human-like expressions: Sounds like a helpful person, not a machine
- Conversational flow: Appropriate use of acknowledgments, questions, and follow-ups
- Varied vocabulary: Avoids repetitive or templated language

MAJOR NATURALNESS VIOLATIONS (score 1):
- Explicit system exposure: "tool call format", "system capabilities", "I'm having trouble with the tool call format"
- Breaking character: Revealing internal workings, mentioning tools/systems directly
- Meta-commentary: "I'll send a message to the user", "let me clarify the system's capabilities"
- Role confusion: "To the user:", "the user hasn't provided", "ask the user for", treating user as third party

DETAILED SCORING SCALE:

Score 5 - PERFECTLY NATURAL (rare, exceptional quality):
- Indistinguishable from a skilled human assistant
- Natural contractions, varied phrasing, authentic warmth
- Perfect conversational flow with genuine helpfulness
- Example: "I'd love to help you find the perfect flight! What city are you departing from?"

Score 4 - VERY NATURAL (good, human-like):
- Sounds like a helpful human with minor imperfections
- Mostly natural language with good conversational flow
- Appropriate friendliness and helpfulness
- Example: "I can help you with that! Could you tell me where you're traveling from?"

Score 3 - NATURAL ENOUGH (acceptable, functional):
- Professional and clear, though somewhat formal
- Gets the job done without being robotic
- May lack warmth but maintains helpfulness
- Example: "I can assist you with flight information. What is your departure city?"

Score 2 - SOMEWHAT UNNATURAL (stiff but functional):
- Overly formal or templated language
- Functional but lacks conversational flow
- Sounds more like a system than a person
- Example: "I will process your flight request. Please provide departure location."

Score 1 - VERY ROBOTIC/SYSTEM EXPOSURE (major violations):
- Exposes technical details or system workings
- Breaks character or mentions internal processes
- Difficult to understand or completely artificial
- Example: "I'm having trouble with the tool call format"

EXAMPLES OF NATURAL vs UNNATURAL PHRASING:

NATURAL (Score 4-5):
- "I'd be happy to help you find flights!"
- "Great question! Let me see what options are available."
- "That sounds perfect. What date were you thinking?"
- "Sure thing! Where are you looking to travel?"

UNNATURAL (Score 2-3):
- "I will assist you with flight information retrieval."
- "Please provide the required parameters for processing."
- "Your request has been received and will be processed."
- "I need to obtain additional data points to proceed."

SYSTEM VIOLATIONS (Score 1):
- "I'm having trouble with the tool call format"
- "Let me clarify the system's capabilities"
- "I'll send a message to the user to ask for more information"
- "The tool requires the origin city parameter"

OUTPUT FORMAT: Return only a JSON object:
{"score": 1-5}"""

    dialogue_text = "CONVERSATION:\n"
    for msg in completion:
        if msg["role"] in ["user", "assistant"]:
            content = msg["content"]
            if msg["role"] == "user" and content.startswith("<result>"):
                dialogue_text += f"SYSTEM: [Tool result]\n"
            else:
                dialogue_text += f"{msg['role'].upper()}: {content}\n"
    
    user_prompt = f"{dialogue_text}\n Evaluate the naturalness of the assistant's responses:"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = eval_llm.chat(messages)
        response_lines = response.strip().split("\n")
        json_line = response_lines[0].strip()
        result = json.loads(json_line)
        return int(result.get("score", 3))
    except Exception as e:
        return 3


# convert LLM score into reward
def map_score_to_range(score, min_score, max_score, min_reward, max_reward):
    if score <= min_score:
        return min_reward
    if score >= max_score:
        return max_reward
    
    ratio = (score - min_score) / (max_score - min_score)
    return min_reward + ratio * (max_reward - min_reward)


# Class for turn-level naturalness rewards
class TurnLevelRewardFunctionNaturalness:

    def __init__(self, config, eval_llm=None, tokenizer=None):
        self.config = config
        self.eval_llm = eval_llm
        self.tokenizer = tokenizer
        self.current_step = 0
        self.__name__ = "TurnLevelRewardFunctionNaturalness"
        self._last_reward_breakdowns = []
    

    # Get reward
    def __call__(self, prompts, completions, raw_datapoint=None, **kwargs):
        rewards = []
        self._last_reward_breakdowns = []
        
        for i, sim_conv in enumerate(completions):
            messages = self._convert_sim_conv_to_messages(sim_conv)
            
            total_reward, reward_breakdown = self._compute_single_completion_reward_detailed(messages, **kwargs)
            capped_reward = max(total_reward, self.config.conciseness_cap_negative)
            rewards.append(capped_reward)
            
            self._last_reward_breakdowns.append({
                "total_reward": total_reward,
                "capped_reward": capped_reward,
                "breakdown": reward_breakdown
            })
        
        self.current_step += 1
        return rewards
    

    # Convert sim to right format
    def _convert_sim_conv_to_messages(self, sim_conv):
        messages = []
        messages.append({"role": "system", "content": ""})
        
        for turn in sim_conv:
            user_msg = turn.get("user", "").strip()
            agent_msg = turn.get("agent", "").strip()
            
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            
            if agent_msg:
                messages.append({"role": "assistant", "content": agent_msg})
        
        return messages
    

    # Reward breakdown for each turn
    def _compute_single_completion_reward_detailed(self, completion, **kwargs):
        total_reward = 0.0
        reward_components = {}
        
        assistant_messages = self._extract_user_facing_messages_by_turn(completion)
        
        if not assistant_messages:
            return 0.0, {
                "assistant_turns_count": 0, 
                "components": {}, 
                "cap_negative": self.config.conciseness_cap_negative
            }
        
        conciseness_reward, conciseness_breakdown = compute_conciseness_rewards(
            assistant_messages, self.config, self.tokenizer
        )
        total_reward += conciseness_reward
        reward_components["conciseness"] = {
            "value": conciseness_reward,
            "breakdown": conciseness_breakdown,
            "description": "Cost per turn and overlong penalties"
        } 
        
        reward_breakdown = {
            "assistant_turns_count": len(assistant_messages),
            "components": reward_components,
            "cap_negative": self.config.conciseness_cap_negative,
            "total_uncapped": total_reward
        }
        
        return total_reward, reward_breakdown
    

    # Get asssistant message
    def _extract_user_facing_messages_by_turn(self, completion):
        
        assistant_messages = []
        
        for msg in completion:
            if msg.get("role") == "assistant" and msg.get("content"):
                assistant_messages.append(msg["content"])
        
        return assistant_messages
    

    # Check if assistant message is a JSON tool call
    def _is_tool_call(self, message):
        
        message = message.strip()
        if not message:
            return False
        if message.startswith("{") and message.endswith("}"):
            try:
                parsed = json.loads(message)
                if isinstance(parsed, dict) and any(key in parsed for key in ["tool", "tool_name", "function", "name", "action"]):
                    return True
            except json.JSONDecodeError:
                pass
        
        return False

# Class for terminal/episodic naturalness rewards
class EpisodicNaturalnessRewardFunction:
    
    def __init__(self, config, eval_llm=None, tokenizer=None):
        self.config = config
        self.eval_llm = eval_llm
        self.tokenizer = tokenizer
        self.current_step = 0
        self.__name__ = "EpisodicNaturalnessRewardFunction"
        self._last_reward_breakdowns = []
    

    # Compute reward
    def __call__(self, prompts, completions, raw_datapoint=None, **kwargs):
        
        rewards = []
        self._last_reward_breakdowns = []
        
        for i, sim_conv in enumerate(completions):
            messages = self._convert_sim_conv_to_messages(sim_conv)
            total_reward, reward_breakdown = self._compute_episodic_naturalness_reward(messages, **kwargs)
            capped_reward = max(min(total_reward, self.config.naturalness_cap_positive), 
                              self.config.naturalness_cap_negative)
            
            rewards.append(capped_reward)
            
            self._last_reward_breakdowns.append({
                "total_reward": total_reward,
                "capped_reward": capped_reward,
                "breakdown": reward_breakdown
            })
        
        self.current_step += 1
        return rewards
    

    # Compute episodic reward 
    def _compute_episodic_naturalness_reward(self, completion, **kwargs):

        total_reward = 0.0
        breakdown = {"helpfulness": {},"naturalness": {}}
        
        if self.config.enable_helpfulness and self.eval_llm:
            helpfulness_reward, helpfulness_breakdown = compute_helpfulness_rewards(
                completion, self.eval_llm, self.config
            )
            total_reward += helpfulness_reward
            breakdown["helpfulness"] = helpfulness_breakdown
        
        if self.config.enable_naturalness and self.eval_llm:
            naturalness_reward, naturalness_breakdown = compute_naturalness_rewards(
                completion, self.eval_llm, self.config
            )
            total_reward += naturalness_reward
            breakdown["naturalness"] = naturalness_breakdown
        
        breakdown["total_uncapped"] = total_reward
        return total_reward, breakdown
    

    # Convert sim to messages format
    def _convert_sim_conv_to_messages(self, sim_conv):
        messages = []
        
        for turn in sim_conv:
            if "user" in turn and turn["user"].strip():
                if not turn["user"].startswith("<result>"):
                    messages.append({
                        "role": "user",
                        "content": turn["user"]
                    })
            
            if "agent" in turn and turn["agent"].strip():
                messages.append({
                    "role": "assistant", 
                    "content": turn["agent"]
                })
        
        return messages

# Create turn-level naturalness reward functions
def create_naturalness_reward_function(config_name="rew_cfg_2_naturalness", config_dict=None, eval_llm=None, tokenizer=None):
    
    config = TurnRewardConfigNaturalness(config_dict)
    reward_func = TurnLevelRewardFunctionNaturalness(config, eval_llm, tokenizer)
    reward_func.__name__ = f"naturalness_reward_{config_name}"
    
    return reward_func

# Combination of different reward functions
def create_episodic_naturalness_reward_function(config_name="rew_cfg_2_naturalness", config_dict=None, eval_llm=None, tokenizer=None):

    config = TurnRewardConfigNaturalness(config_dict)
    return EpisodicNaturalnessRewardFunction(config, eval_llm, tokenizer)

# Create turn-level reward function
def make_naturalness_reward_func(config_name="rew_cfg_2_naturalness", config_dict=None, eval_llm=None, tokenizer=None):
    return create_naturalness_reward_function(config_name, config_dict, eval_llm, tokenizer)

# Create episodic reward function
def make_episodic_naturalness_reward_func(config_name="rew_cfg_2_naturalness", config_dict=None, eval_llm=None, tokenizer=None):
    return create_episodic_naturalness_reward_function(config_name, config_dict, eval_llm, tokenizer)