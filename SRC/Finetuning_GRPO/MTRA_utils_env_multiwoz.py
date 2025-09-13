from __future__ import annotations
import json
import torch
from .MTRA_utils_env import SGDEnvAdapter
from .MTRA_utils_multiwoz import create_multiwoz_simulator_function

# Extend env from SGD base to work with MultiWOZ
class MultiWOZEnvAdapter(SGDEnvAdapter):

    def __init__(self, *, tokenizer, assistant_models=None, user_models=None, schemas=None, system_prompt_template="", max_turns=8, benchmark="MultiWOZ"):
        super().__init__(tokenizer=tokenizer, assistant_models=assistant_models, user_models=user_models, schemas=schemas, system_prompt_template=system_prompt_template, max_turns=max_turns)
        self.benchmark = benchmark

    def get_rubric(self, **kwargs):
        if self.benchmark == "MultiWOZ":
            return ["inform", "success", "book"]
        else:
            return super().get_rubric(**kwargs)
    
    def generate(self, *, prompts, llm, sampling_params, simulator, prompt_to_dp, build_initial_user_text, policy_seed, num_generations, benchmark):
        return super().generate(prompts=prompts, llm=llm, sampling_params=sampling_params, simulator=simulator, prompt_to_dp=prompt_to_dp, build_initial_user_text=build_initial_user_text, policy_seed=policy_seed, num_generations=num_generations, benchmark=benchmark)