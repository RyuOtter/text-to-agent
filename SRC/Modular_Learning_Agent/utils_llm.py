from peft import PeftModel, LoraConfig, get_peft_model
import os
import time
import yaml
import openai
from openai import OpenAI, RateLimitError
import groq
from groq import APIConnectionError, RateLimitError as GroqRateLimitError
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline,BitsAndBytesConfig
import torch
import random
import numpy as np

# Model class used for calling LLM models either through API or locally after download from HuggingFace
class LLMModel:

    # Initialize model depending on provider
    def __init__(self, provider, model_name, temperature=0.7, max_tokens=2048, existing_model=None, existing_tokenizer=None, mtra_llm=None, mtra_sampling_params=None, lora_adapter_path=None):
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.lora_adapter_path = lora_adapter_path
        self._load_api_keys()

        # Unsloth is no longer used, was part of a previous GRPO implementation
        """
        if self.provider == "unsloth" and existing_model is not None and existing_tokenizer is not None:
            self.model = existing_model
            self.tokenizer = existing_tokenizer
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if getattr(self.model.config, "pad_token_id", None) is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            return
        """

        if self.provider == "mtra_vllm":
            self.tokenizer = existing_tokenizer
            self.mtra_llm = mtra_llm
            self.mtra_sampling_params = mtra_sampling_params
            return

        if self.provider == "openai":
            openai.api_key = self.api_keys.get("openai")

        elif self.provider == "gemini":
            genai.configure(api_key=self.api_keys.get("gemini"))

        elif self.provider == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name,device_map="auto",torch_dtype=torch.bfloat16)

            if self.lora_adapter_path:
                self.model = PeftModel.from_pretrained(self.model, self.lora_adapter_path)

            self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        elif self.provider == "groq":
            self.groq_client = groq.Groq(api_key=self.api_keys.get("groq"))
            
        elif self.provider == "groq_improver":
            self.groq_client = groq.Groq(api_key=self.api_keys.get("groq_improver"))

        elif self.provider == "unsloth":
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(self.model_name,max_seq_length=8192,dtype=None,load_in_4bit=False,device_map="auto")
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if getattr(self.model.config, "pad_token_id", None) is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

        else:
            raise ValueError("Unsupported model provider")

    # Load API keys from config
    def _load_api_keys(self):
        cfg = {}
        config_paths = ["config.yaml",os.path.join(os.path.dirname(__file__), "config.yaml")]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f) or {}
                cfg = config.get("api_keys", {}) or {}
                break
        self.api_keys = {
            "openai": os.getenv("OPENAI_API_KEY") or cfg.get("openai"),
            "gemini": os.getenv("GEMINI_API_KEY") or cfg.get("gemini"),
            "groq": os.getenv("GROQ_API_KEY") or cfg.get("groq"),
            "groq_improver": os.getenv("GROQ_IMPROVER_API_KEY") or cfg.get("groq_improver"),
            "huggingface": os.getenv("HUGGINGFACE_API_KEY") or cfg.get("huggingface"),
        }

    # Generate an LLM response
    def chat(self, messages, return_usage=False):
        if self.provider == "openai":
            response = self._chat_openai(messages)
            content = response.choices[0].message.content
            if return_usage:
                return content, response.usage.model_dump()
            return content

        elif self.provider == "gemini":
            return self._chat_gemini(messages)

        elif self.provider == "groq" or self.provider == "groq_improver":
            response = self._chat_groq(messages)
            if return_usage:
                estimated_tokens = len(response.split()) * 1.3
                usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": int(estimated_tokens),
                    "total_tokens": int(estimated_tokens),
                }
                return response, usage
            return response

        elif self.provider == "huggingface":
            return self._chat_huggingface(messages)

        elif self.provider == "unsloth":
            text, meta = self.chat_with_metadata(messages)
            if return_usage:
                usage = {
                    "prompt_tokens": int(meta["prompt_input_ids"].numel()),
                    "completion_tokens": int(meta["generated_ids"].numel()),
                    "total_tokens": int(meta["prompt_input_ids"].numel() + meta["generated_ids"].numel()),
                }
                return text, usage
            return text

        elif self.provider == "mtra_vllm":
            chat_text = self._messages_to_prompt_with_template(messages)
            outputs = self.mtra_llm.generate([chat_text], self.mtra_sampling_params)
            try:
                return outputs[0].outputs[0].text
            except Exception:
                return str(outputs[0].outputs[0])

        else:
            raise NotImplementedError("Provider not implemented")

    # Set seed for reproducibility in local models
    @staticmethod
    def _set_seed(seed):
        if seed is None:
            return
        random.seed(seed)
        if np is not None:
            np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Convert messages to prompt 
    def _messages_to_prompt(self, messages):
        if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            except Exception:
                pass

        if len(messages) == 1 and messages[0]["role"] == "system":
            return messages[0]["content"]
        return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    # Convert messages to prompt with Llama template
    def _messages_to_prompt_with_template(self, messages):
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return self._messages_to_prompt(messages)
    
    # Get device model is on
    def _model_device(self):
        try:
            return next(self.model.parameters()).device
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LLM response generation for OpenAI
    def _chat_openai(self, messages):
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                client = OpenAI(api_key=self.api_keys.get("openai"))
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response
            except RateLimitError as e:
                if attempt < max_attempts - 1:
                    backoffs = [65, 80, 100, 120]
                    sleep_s = backoffs[attempt] if attempt < len(backoffs) else 120
                    time.sleep(sleep_s)
                    continue
                raise
            except Exception as e:
                if attempt < max_attempts - 1:
                    time.sleep(min(2 ** attempt, 10))
                    continue
                raise

    # LLM response generation for Gemini
    def _chat_gemini(self, messages):
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.text

    # LLM response generation for Groq
    def _chat_groq(self, messages):
        formatted_messages = []
        for m in messages:
            if m["role"] == "function":
                tool_name = m.get("name", "unknown_tool")
                tool_result = m.get("content", "")
                formatted_messages.append({
                    "role": "system", 
                    "content": f"Function call result from {tool_name}: {tool_result}"
                })
            else:
                formatted_messages.append({"role": m["role"], "content": m["content"]})
        
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.model_name,
                    messages=formatted_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except groq.RateLimitError as e:
                if attempt < max_attempts - 1:
                    backoffs = [70, 90, 110, 130]
                    sleep_s = backoffs[attempt] if attempt < len(backoffs) else 130
                    print("Groq rate limit hit, waiting before retry")
                    time.sleep(sleep_s)
                    continue
                raise
            except Exception as e:
                if attempt < max_attempts - 1:
                    time.sleep(min(2 ** attempt, 10))
                    continue
                raise

    # LLM response generation via local model from HuggingFace
    def _chat_huggingface(self, messages):
        prompt = self._messages_to_prompt(messages)
        do_sample = self.temperature > 0.0
        out = self.pipeline(
            prompt,
            max_new_tokens=self.max_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            top_p=0.9 if do_sample else None,
        )
        return out[0]["generated_text"].split(prompt, 1)[-1].strip()