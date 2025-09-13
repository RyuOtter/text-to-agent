from __future__ import annotations
import os
import yaml
from datasets import Dataset
from Modular_Learning_Agent.utils_sim import Agent
from Modular_Learning_Agent.utils_tools import get_tools

# Function to load YAML config
def load_yaml_config(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data

# Function to build assistant prompt with tools and conv history
def build_first_assistant_prompt(*, datapoint, assistant_prompt_template, schemas, assistant_models, benchmark, user_agent=None):
    tmp_asst = Agent(name="assistant", role="assistant", system_prompt=assistant_prompt_template, model=assistant_models, tools=None)
    tmp_asst.set_tools(get_tools(benchmark=benchmark, datapoint=datapoint, schemas=schemas, dynamic=True), schemas=schemas)
    u0 = extract_first_user_text(datapoint)

    conv_hist = f"user: {u0}\n"
    return tmp_asst.prompt.replace("{conversation}", conv_hist.strip())

# Function to get first user message from SGD/MultiWOZ datapoint
def extract_first_user_text(dp):
    for key in ("turns", "dialogue", "utterances", "messages", "conversation"):
        seq = dp.get(key)
        if isinstance(seq, list):
            for t in seq:
                if not isinstance(t, dict):
                    continue

                if isinstance(t.get("user"), str) and t["user"].strip():
                    return t["user"].strip()

                speaker = (t.get("speaker") or t.get("role") or t.get("author") or "").strip().lower()
                text = (t.get("utterance") or t.get("text") or t.get("content") or t.get("message") or t.get("value") or "")
                if speaker in {"user", "u", "customer", "client"} and isinstance(text, str) and text.strip():
                    return text.strip()

    for key in ("initial_user_utterance", "first_user_utterance", "start_user", "first_user", "initial_utterance"):
        val = dp.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    for key in ("instruction", "goal", "task", "query"):
        val = dp.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

# Transform datapoints into training data format for the env trainer
def build_dataset_and_maps(dialogs, assistant_prompt, assistant_models, schemas, benchmark):
    prompts = []
    prompt_to_dp = {}
    for idx, d in enumerate(dialogs):
        ptxt = build_first_assistant_prompt(datapoint=d, assistant_prompt_template=assistant_prompt, schemas=schemas, assistant_models=assistant_models, benchmark=benchmark)
        prompts.append(ptxt)
        prompt_to_dp[ptxt] = d

    dataset = Dataset.from_dict({"prompt": prompts, "raw_datapoint": dialogs})
    return dataset, prompt_to_dp
