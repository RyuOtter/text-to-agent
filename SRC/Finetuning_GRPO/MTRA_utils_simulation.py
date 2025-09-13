from __future__ import annotations
import re, json, torch
from Modular_Learning_Agent.sgd_data_utils import prepare_goals_str
from Modular_Learning_Agent.utils_tools import get_tools
from Modular_Learning_Agent.utils_sim import Agent
from Modular_Learning_Agent.multiwoz_data_utils import prepare_goals_string
from verifiers.imports import SamplingParams

# Class used to store metadata needed for GRPO computations per turn
class AssistantTurn:
    def __init__(self, prompt_input_ids, generated_ids, old_logprobs):
        self.prompt_input_ids = prompt_input_ids
        self.generated_ids = generated_ids
        self.old_logprobs = old_logprobs

    def length(self):
        return int(self.generated_ids.numel())

# Legacy from previous unsloth implementation
class RLRecorder:
    def __init__(self, turns=None, meta=None):
        self.turns = turns if turns is not None else []
        self.meta = meta if meta is not None else {}

    def add_turn(self, prompt_input_ids, generated_ids, old_logprobs):
        self.turns.append(
            AssistantTurn(
                prompt_input_ids=prompt_input_ids.detach().cpu(),
                generated_ids=generated_ids.detach().cpu(),
                old_logprobs=old_logprobs.detach().cpu().to(torch.float32),
            )
        )
        del prompt_input_ids, generated_ids, old_logprobs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def num_emissions(self):
        return len(self.turns)

    def num_generated_tokens(self):
        return sum(t.length() for t in self.turns)

    def concat_generated_ids(self):
        if not self.turns:
            return torch.empty(0, dtype=torch.long)
        return torch.cat([t.generated_ids for t in self.turns], dim=0)

    def concat_old_logprobs(self):
        if not self.turns:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat([t.old_logprobs for t in self.turns], dim=0)

    def sum_old_logprobs(self):
        if not self.turns:
            return 0.0
        return float(sum(t.old_logprobs.sum().item() for t in self.turns))

# Legacy from previous unsloth implementation
class RLGroup:
    def __init__(self, trajectories=None, meta=None):
        self.trajectories = trajectories if trajectories is not None else []
        self.meta = meta if meta is not None else {}

    def new_recorder(self, **meta):
        rec = RLRecorder(meta=meta)
        self.trajectories.append(rec)
        return rec

    def __len__(self):
        return len(self.trajectories)

    def sum_old_logprobs_batch(self):
        if not self.trajectories:
            return torch.empty(0, dtype=torch.float32)
        vals = [t.sum_old_logprobs() for t in self.trajectories]
        return torch.tensor(vals, dtype=torch.float32)

    def total_tokens(self):
        return sum(t.num_generated_tokens() for t in self.trajectories)

# Simulator function for GRPO
def create_simulator_function(user_llm, eval_llm, max_turns, user_prompt, assistant_prompt):
    def simulator(**kwargs):

        user_agent = Agent(
            name="user",
            system_prompt=user_prompt,
            model=kwargs.get("user_models") or {"dialogue": user_llm},
            role="user"
        )
        assistant_agent = Agent(
            name="assistant",
            system_prompt=assistant_prompt,
            model=kwargs.get("assistant_models") or {"dialogue": kwargs.get("llm")},
            role="assistant"
        )
        
        dp = kwargs.get("datapoint") or kwargs.get("dp")
        schemas = kwargs.get("schemas")
        policy_seed = kwargs.get("policy_seed")
        benchmark = kwargs.get("benchmark", "SGD")
        

        res = single_simulation_rl(
            benchmark=benchmark,
            individual_dialog=dp,
            user=user_agent,
            assistant=assistant_agent,
            max_iterations=max_turns,
            schemas=schemas,
            policy_seed=policy_seed,
        )
        sim_conv, tool_logs, recorder, assistant_only = res
        return sim_conv, tool_logs, recorder, assistant_only
        
    return simulator

# Get first user message for MultiWOZ
def extract_first_user_text_multiwoz(dp):
    log = dp.get("log", [])
    first_turn = log[0]
    utterance = first_turn.get("text", "").strip()
    
    return utterance

# Get first user message for SGD
def extract_first_user_text_sgd(dp):
    turns = dp.get("turns")
    if isinstance(turns, list) and turns:
        for t in turns:
            if not isinstance(t, dict):
                continue
            
            if isinstance(t.get("user"), str) and t["user"].strip():
                return t["user"].strip()

            speaker = (t.get("speaker") or t.get("role") or t.get("author") or "").strip().lower()
            text = (
                t.get("utterance")
                or t.get("text")
                or t.get("content")
                or t.get("message")
                or t.get("value")
                or ""
            )
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

# Extract first user message
def create_build_initial_user_text_function():
    def build_initial_user_text(dp):
        if "log" in dp and "turns" not in dp:
            return extract_first_user_text_multiwoz(dp)
        else:
            return extract_first_user_text_sgd(dp)
        
    return build_initial_user_text

# Setup enviornment rollout function
def setup_environment_generate(env, simulator, prompt_to_dp, build_initial_user_text, grpo_cfg, data_cfg, training_args, logged_generate_func):
    env.generate = lambda **gen_kwargs: logged_generate_func(
        env,
        prompts=gen_kwargs["prompts"],
        llm=gen_kwargs["llm"],
        sampling_params=gen_kwargs["sampling_params"],
        simulator=simulator,
        prompt_to_dp=prompt_to_dp,
        build_initial_user_text=build_initial_user_text,
        policy_seed=int(grpo_cfg.get("base_policy_seed", 1337)),
        num_generations=training_args.num_generations,
        benchmark=data_cfg.get("benchmark", "SGD"),
    )

# Copy over weights from fine-tuned model to vLLM for on-policy training
def enable_vllm_weight_reloading_with_confirmation(trainer):
    if hasattr(trainer, "_move_model_to_vllm"):
        original_move_func = trainer._move_model_to_vllm
        
        def confirmed_move_model_to_vllm():
            print("On-policy training: LoRA adapters merged into base model")
            return original_move_func()
        
        trainer._move_model_to_vllm = confirmed_move_model_to_vllm

# Off-policy switch
def disable_vllm_weight_reloading(trainer):
    if hasattr(trainer, "_move_model_to_vllm"):
        trainer._move_model_to_vllm = lambda: None

# Setup for vLMM
def setup_vllm_sampling_params(trainer, assistant_temperature, assistant_max_tokens, max_seq_len):
    try:
        if hasattr(trainer, "sampling_params"):
            old_params = getattr(trainer, "sampling_params", None)
            trainer.sampling_params = SamplingParams(
                temperature=assistant_temperature,
                max_tokens=min(assistant_max_tokens, max_seq_len),
                top_p=getattr(old_params, "top_p", 0.9) if old_params else 0.9,
                seed=getattr(old_params, "seed", None) if old_params else None,
                logprobs=1 
            )
        else:
            print("No trainer.sampling_params")
    except Exception as e:
        print(f"Override failed for vLLM: {e}")

# Cleaning text for JSON identification
def _strip_fences(s):
    if not isinstance(s, str): return ""
    t = s.strip()
    t = re.sub(r"^\s*```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()

# JSON identification check
def _looks_like_tool_json(s):
    try:
        obj = json.loads(_strip_fences(s))
        return isinstance(obj, dict) and "tool" in obj and isinstance(obj.get("args"), dict)
    except Exception:
        return False

# Simulation function to simulate conversation with storing needed for GPRO
def single_simulation_rl(benchmark, individual_dialog, user, assistant, *, max_iterations=10, schemas=None, user_mode="llm", policy_seed=None, user_seed=None):

    prev_flag = getattr(assistant, "grpo_training", False)
    assistant.grpo_training = True
    if hasattr(assistant, "_reset_action_trace"):
        assistant._reset_action_trace()
    else:
        assistant.action_trace = []

    assistant.tool_log = []

    if benchmark == "SGD":
        task_goal = prepare_goals_str(individual_dialog)
    elif benchmark == "MultiWOZ":
        goal_messages = individual_dialog.get("goal", {}).get("message", [])
        task_goal = prepare_goals_string(goal_messages) if goal_messages else f"Dialog {individual_dialog.get('dialog_id', 'unknown')}"
    else:
        raise ValueError("Wrong benchmark")
    
    user.include_user_goal(task_goal)

    if benchmark == "SGD":
        assistant.set_tools(
            get_tools(benchmark=benchmark, datapoint=individual_dialog, schemas=schemas, dynamic=True),
            schemas=schemas
        )
    elif benchmark == "MultiWOZ":
        if not hasattr(assistant, "tools") or not assistant.tools:
            assistant.set_tools(
                get_tools(benchmark=benchmark, datapoint=None, schemas=None, dynamic=False)
            )

    simulated_conversation = []
    turn_idx = 1
    rl_recorder = None

    for _ in range(max_iterations):

        if user_mode == "ground_truth":
            user_response = user(simulated_conversation, individual_dialog)
        else:
            user_response = user(simulated_conversation, policy_seed=user_seed)

        if "Task Completed" in user_response:
            simulated_conversation.append({
                "turn_idx": turn_idx,
                "user": user_response,
                "agent": "Goodbye.",
            })
            break

        simulated_conversation.append({"turn_idx": turn_idx, "user": user_response})
        assistant_response = assistant(
            simulated_conversation,
            rl_recorder=None,
            policy_seed=policy_seed,
        )

        simulated_conversation[-1]["agent"] = assistant_response
        turn_idx += 1

    function_logs = assistant.tool_log
    assistant.tool_log = []

    parts = []
    for i, (kind, text) in enumerate(getattr(assistant, "action_trace", []), start=1):
        body = (text or "").strip()
        if not body:
            continue
        if kind.startswith("TOOL_CALL"):
            parts.append(f"<A{i}_{kind}>{_strip_fences(body)}</A{i}_{kind}>")
        else:
            parts.append(f"<A{i}_{kind}>{body}</A{i}_{kind}>")
    assistant_only = "\n".join(parts)
    assistant.grpo_training = prev_flag
    assistant.action_trace = []

    return simulated_conversation, function_logs, None, assistant_only
