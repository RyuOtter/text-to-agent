from __future__ import annotations
import json
import torch
from datasets import Dataset
from verifiers.envs.environment import Environment
from Modular_Learning_Agent.utils_sim import Agent
from Modular_Learning_Agent.utils_tools import get_tools
from Modular_Learning_Agent.utils_llm import LLMModel

# Env for SGD
class SGDEnvAdapter(Environment):

    # Initialize env with model etc.
    def __init__(self, *, tokenizer, assistant_models=None, user_models=None, schemas=None, system_prompt_template="", max_turns=8):
        super().__init__()
        self.tokenizer = tokenizer
        self.assistant_models = assistant_models
        self.user_models = user_models
        self.schemas = schemas
        self.system_prompt_template = system_prompt_template
        self.max_turns = max_turns

    def get_dataset(self, **kwargs):
        return getattr(self, "dataset", None)

    def get_eval_dataset(self, **kwargs):
        return getattr(self, "eval_dataset", None)

    def get_rubric(self, **kwargs):
        return []

    # Function used to generate roll outs for training
    def generate(self, *, prompts, llm, sampling_params, simulator, prompt_to_dp, build_initial_user_text, policy_seed, num_generations, benchmark):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        all_ids = []
        all_masks = []
        all_sim_conv = []
        all_assistant_only = []
        all_tool_logs = []
        all_raw_datapoint = []

        for prompt in prompts:
            dp = prompt_to_dp[prompt]
            dialog_id = dp.get("dialogue_id", "unknown")
            initial_user_text = build_initial_user_text(dp)

            for i in range(num_generations):
                assistant_seed = policy_seed + i
                onpolicy_assistant = LLMModel(provider="mtra_vllm", model_name="trainer_vllm", existing_tokenizer=self.tokenizer, mtra_llm=llm, mtra_sampling_params=sampling_params)
                assistant_models = dict(self.assistant_models) if self.assistant_models else {}
                assistant_models["dialogue"] = onpolicy_assistant
                assistant_models["correction"] = onpolicy_assistant

                sim_conv, tool_logs, rl_recorder, assistant_only = simulator(datapoint=dp, assistant_models=assistant_models, user_models=self.user_models, schemas=self.schemas, max_turns=self.max_turns, benchmark=benchmark, policy_seed=assistant_seed)
            
                actions_text = assistant_only

                all_sim_conv.append(sim_conv)
                all_assistant_only.append(assistant_only)
                all_tool_logs.append(tool_logs)
                all_raw_datapoint.append(dp)

                completion_text = actions_text if isinstance(actions_text, str) else ""
                
                tokenized = self.tokenizer(completion_text, add_special_tokens=False, return_tensors=None)
                ids = tokenized["input_ids"] if isinstance(tokenized["input_ids"], list) else tokenized["input_ids"][0]
                if isinstance(ids, list) and ids and isinstance(ids[0], list):
                    ids = ids[0]
                mask = [1] * len(ids)
                all_ids.append(ids)
                all_masks.append(mask)
        
        return {"ids": all_ids, "messages": all_sim_conv, "mask": all_masks, "assistant_only": all_assistant_only, "tool_logs": all_tool_logs, "raw_datapoint": all_raw_datapoint}