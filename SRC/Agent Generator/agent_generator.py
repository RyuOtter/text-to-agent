import json
from dataclasses import dataclass
from openai import OpenAI
from config import OPENAI_API_KEY

# Definition of used classes
@dataclass
class TaskDescription:
    objective: str

@dataclass
class ICLAgent:
    system_prompt: str
    
    def save(self, filepath):
        with open(filepath, "w") as f:
            json.dump({
                "system_prompt": self.system_prompt
            }, f, indent=2)

class OpenAILLM:
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def generate(self, prompt, max_new_tokens=256, temperature=0.7, system_message=None):
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content

# Role and task generating agent
class RoleTaskGenerator:
    def __init__(self, llm):
        self.llm = llm
    
    def generate(self, task):
        system_message = "You are an expert in defining roles and task descriptions in system prompts for AI agents."
        prompt = f"""You are given the following goal:

Goal: {task.objective}

Please write a clear role and task description for an AI agent that will perform this task.

IMPORTANT: 
- Output ONLY the role and task description text, no introductory words
- Start with "You are a helpful assistant"
- Be specific about what the agent does and its primary function
- Keep it concise but informative (2-3 sentences maximum)
- Focus on the agent's identity and core responsibility

Examples:
- You are a helpful assistant specializing in sentiment analysis of customer reviews. Your primary function is to analyze text and determine whether the sentiment expressed is positive, negative, or neutral.
- You are a helpful assistant specializing in language translation. Your role is to accurately translate text from one language to another while preserving meaning and context."""
        return self.llm.generate(prompt, max_new_tokens=4000, temperature=0.8, system_message=system_message)

# Rules generating agent
class RulesGenerator:
    def __init__(self, llm):
        self.llm = llm
    
    def generate(self, task):
        system_message = "You are an expert in creating behavioral rules for AI agents."
        prompt = f"""You are given the following goal:

Goal: {task.objective}

Please write 2-4 specific behavioral rules that will guide an AI agent in performing this task effectively. You should write the rules in the system prompt format.

IMPORTANT:
- Output ONLY the rules section, no introductory words
- Start with "RULES:" on its own line
- Use bullet points for each rule
- Each rule should be one clear, actionable sentence
- Use action verbs (analyze, identify, provide, ensure, etc.)
- Be specific about how the agent should behave
- Focus on process and quality standards

Examples:
RULES:
- Analyze the text carefully to identify emotional indicators and context clues.
- Classify sentiment as positive, negative, or neutral based on the overall tone.
- Provide a brief justification for your classification."""
        return self.llm.generate(prompt, max_new_tokens=4000, temperature=0.8, system_message=system_message)

# Examples generating agent
class ExamplesGenerator:
    def __init__(self, llm):
        self.llm = llm
    
    def generate(self, task):
        system_message = "You are an expert in creating in-context learning examples for AI agents."
        prompt = f"""You are given the following goal:

Goal: {task.objective}

Please create 2-3 clear examples that demonstrate ideal input-output patterns for this task.

IMPORTANT:
- Output ONLY the examples section, no introductory words
- Start with "EXAMPLES:" on its own line
- Use "Input:" and "Output:" format for each example
- Show realistic, diverse inputs that cover different aspects of the task
- Provide ideal outputs that demonstrate the expected quality and format
- Keep examples concise but informative
- Make sure outputs show the complete expected response pattern

Examples:
EXAMPLES:
Task: Analyze customer reviews and determine if they are positive, negative, or neutral
Input: "This restaurant has amazing food and great service!"
Output: Sentiment: Positive. The review contains positive emotional indicators like "amazing" and "great" which clearly express satisfaction with both food quality and service.

Input: "The product broke after two days and customer service was unhelpful."
Output: Sentiment: Negative. The review expresses dissatisfaction through negative experiences with product durability and poor customer service interaction."""
        return self.llm.generate(prompt, max_new_tokens=4000, temperature=0.8, system_message=system_message)

# Revision agent that combines all components into one system prompt
class RevisionAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def generate(self, task, role_task, rules, examples):
        system_message = "You are an expert in creating cohesive, well-structured system prompts for AI agents. Your role is to merge different components into a single, clear, and actionable system prompt."
        prompt = f"""You are given a task objective and three generated components that need to be merged into a single, cohesive system prompt.

TASK OBJECTIVE: {task.objective}

ROLE & TASK DESCRIPTION:
{role_task}

{rules}

{examples}

Your task is to merge these components into a single, well-structured system prompt.

REQUIREMENTS:
- Create ONE cohesive system prompt that flows naturally
- Check for and resolve any contradictions between components
- Ensure all instructions use clear action verbs (analyze, identify, provide, classify, etc.)
- Make instructions clearly understandable and unambiguous
- Maintain logical flow from role definition to behavioral rules to examples
- Ensure the examples align with and reinforce the rules
- Ensure that at least one example contain a tool call to the system with the correct JSON format and dummy tool name and arguments
- Keep the prompt concise but comprehensive

ADDITIONAL REQUIREMENTS:
Next to combining the initial components, you MUST include the following critical components in the system prompt:
- The final system prompt MUST contain the exact placeholders for the conversation and the tool list: 
These are your available tools:
{{tool_list}}
This is the conversation so far with the user:
{{conversation}}
- The final system prompt MUST contain contain instructions for tool calling:
At each turn you must either:
1. Reply naturally to the user OR
2. Call a tool to retrieve information or complete an action for the user using this JSON format. These messages are not send to the user but to the system:
{{
  "tool": "tool_name",
  "args": {{
    "arg1": "value1"
  }}
}}
- The final system prompt MUST contain the specification: We are in the year 2019. Today is 2019-03-01, Friday.

OUTPUT FORMAT:
- Output ONLY the final system prompt
- No introductory text, explanations, or meta-commentary
- The prompt should be ready to use as-is for an AI agent
"""
        return self.llm.generate(prompt, max_new_tokens=4000, temperature=0.0, system_message=system_message)

# Final class that combines all individual agents into one multi-agent system
class AgentGenerator:
    
    def __init__(self):
        self.llm = OpenAILLM()
        self.role_task_generator = RoleTaskGenerator(self.llm)
        self.rules_generator = RulesGenerator(self.llm)
        self.examples_generator = ExamplesGenerator(self.llm)
        self.revision_agent = RevisionAgent(self.llm)
    
    def generate_agent(self, task):
        role_task = self.role_task_generator.generate(task)
        rules = self.rules_generator.generate(task)
        examples = self.examples_generator.generate(task)
        final_system_prompt = self.revision_agent.generate(task, role_task, rules, examples)
        
        return ICLAgent(system_prompt=final_system_prompt)
    
    def create_callable_agent(self, agent, task):
        def agent_function(history, **kwargs):
            system_prompt = agent.system_prompt
            conversation = ""
            for msg in history:
                role = msg["role"].capitalize()
                conversation += f"{role}: {msg["content"]}\n"
            user_prompt = conversation.strip()
            
            temperature = kwargs.get("temperature", 0.7)
            max_new_tokens = kwargs.get("max_new_tokens", 150)
            return self.llm.generate(user_prompt, max_new_tokens=max_new_tokens, temperature=temperature, system_message=system_prompt)
        return agent_function

if __name__ == "__main__":
    # Definition of task description
    task = TaskDescription(
        objective="I want a conversational AI assistant that helps me with all my problems. It should talk to me in plain language, figure out what I’m trying to do, and then get it done end-to-end: search for information, compare my options, and take actions like booking, purchasing, or playing things using whatever tools are available. It should only ask me the essential questions (like dates, location, budget, must-haves, nice-to-haves), show me a small set of clearly explained choices with key trade-offs, and always get my confirmation before spending money or making irreversible changes. It should keep me informed with short updates. If something fails or it cannot do something, tell me what’s possible instead and suggest the best workaround."
    )
    
    
    # Generation of system prompt
    generator = AgentGenerator()
    agent = generator.generate_agent(task)
    
    # Saving of system prompt
    output_filename = "generated_system_prompt.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(agent.system_prompt)