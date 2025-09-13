from typing import Callable, Optional, Union, Any, List, Dict, Tuple

from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    is_wandb_available,
    Trainer,
)
from transformers.utils import is_peft_available
from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import apply_chat_template, maybe_apply_chat_template
from trl.import_utils import is_rich_available
from trl.trainer.utils import pad

from verifiers.envs.environment import Environment
from verifiers.utils.logging_utils import print_prompt_completions_sample

if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[List, List], List[float]]]

class GRPOEnvTrainer(GRPOTrainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            env: Environment,        
            turn_reward_funcs: Union[RewardFunc, List[RewardFunc]],
            outcome_reward_funcs: Union[RewardFunc, List[RewardFunc]],
            turn_reward_weights: Optional[List[float]] = None,
            outcome_reward_weights: Optional[List[float]] = None,
            no_turn_reward: Optional[bool] = None,
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
            peft_config: Optional["PeftConfig"] = None,
            metrics_log_path: Optional[str] = None,
            convo_log_path: Optional[str] = None,
            **kwargs,
    ):
        if not args.use_vllm:
            raise ValueError("vLLM must be enabled for GRPOEnvTrainer")
        if not (callable(turn_reward_funcs) or (isinstance(turn_reward_funcs, list) and all(callable(f) for f in turn_reward_funcs))):
            raise ValueError("turn_reward_funcs must be a function or a list of functions.")
        if not (callable(outcome_reward_funcs) or (isinstance(outcome_reward_funcs, list) and all(callable(f) for f in outcome_reward_funcs))):
            raise ValueError("outcome_reward_funcs must be a function or a list of functions.")

        self.turn_reward_funcs = turn_reward_funcs
        self.outcome_reward_funcs = outcome_reward_funcs
        self.combined_reward_funcs = turn_reward_funcs + outcome_reward_funcs

        self.num_turn_funcs = len(turn_reward_funcs)
        self.num_outcome_funcs = len(outcome_reward_funcs)
        self.num_combined_reward_funcs = len(self.combined_reward_funcs)

        if turn_reward_weights is None:
            self.turn_reward_weights = torch.ones(self.num_turn_funcs)
        else:
            self.turn_reward_weights = torch.tensor(turn_reward_weights)
        if outcome_reward_weights is None:
            self.outcome_reward_weights = torch.ones(self.num_outcome_funcs)
        else:
            self.outcome_reward_weights = torch.tensor(outcome_reward_weights)
        self.combined_reward_weights = torch.cat([self.turn_reward_weights, self.outcome_reward_weights], dim=0)

        self.no_turn_reward = no_turn_reward
        self.metrics_log_path = metrics_log_path
        self.convo_log_path = convo_log_path
        self.actions_log_path = kwargs.pop('actions_log_path', None)
        self.global_generation_counter = 0
        self.logged_conversations = set()
        self.logged_actions = set()

        super().__init__(
            model=model,
            reward_funcs=self.combined_reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )
        
        self.env = env

    def _generate_and_score_completions(
         self, inputs: Dict[str, Union[torch.Tensor, Any]]   
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        
        prompts = [x["prompt"] for x in inputs]
        prompt_ids, prompt_mask = self._prepare_prompt_inputs(inputs)
         
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step
            
        completion_ids, completion_messages, completion_mask, env_assistant_only, env_tool_logs, env_raw_datapoint = self._generate_completions(prompts)

        prompt_completion_ids, attention_mask, logits_to_keep = self._prepare_model_inputs(
            prompt_ids, prompt_mask, completion_ids, completion_mask
        )
        
        old_per_token_logps, ref_per_token_logps = self._compute_logps(
            prompt_completion_ids, attention_mask, logits_to_keep
        )

        env_data = {
            'assistant_only': env_assistant_only,
            'tool_logs': env_tool_logs,
            'raw_datapoint': env_raw_datapoint
        }
        
        turn_rewards_per_func = self._calculate_rewards(
            prompts, completion_messages, self.turn_reward_funcs, inputs, env_data
        )
        outcome_rewards_per_func = self._calculate_rewards(
            prompts, completion_messages, self.outcome_reward_funcs, inputs, env_data
        )
        combined_rewards_per_func = self._calculate_rewards(
            prompts, completion_messages, self.combined_reward_funcs, inputs, env_data
        )

        turn_rewards = (turn_rewards_per_func * self.turn_reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        outcome_rewards = (outcome_rewards_per_func * self.outcome_reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        combined_rewards = (combined_rewards_per_func * self.combined_reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        turn_mean_grouped_rewards, turn_std_grouped_rewards, turn_advantages = self._compute_normalized_advantages(turn_rewards, len(prompts))
        outcome_mean_grouped_rewards, outcome_std_grouped_rewards, outcome_advantages = self._compute_normalized_advantages(outcome_rewards, len(prompts))
        combined_mean_grouped_rewards, combined_std_grouped_rewards, combined_advantages = self._compute_normalized_advantages(combined_rewards, len(prompts))
        
        advantages = outcome_advantages if self.no_turn_reward else combined_advantages
        
        if self.metrics_log_path:
            self._log_training_metrics(
                prompts, completion_messages, 
                turn_rewards_per_func, outcome_rewards_per_func,
                env_raw_datapoint, env_assistant_only, env_tool_logs, inputs
            )

        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        turn_rewards_per_func = turn_rewards_per_func.mean(dim=0)
        for i, reward_func in enumerate(self.turn_reward_funcs):
            reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/turn/{reward_func_name}"].append(turn_rewards_per_func[i].item())
            
        outcome_rewards_per_func = outcome_rewards_per_func.mean(dim=0)
        for i, reward_func in enumerate(self.outcome_reward_funcs):
            reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/outcome/{reward_func_name}"].append(outcome_rewards_per_func[i].item())

        self._metrics[mode]["reward/turn"].append(turn_rewards.mean().item())
        self._metrics[mode]["reward/outcome"].append(outcome_rewards.mean().item())
        self._metrics[mode]["reward/combined"].append(combined_rewards.mean().item())
        self._metrics[mode]["reward_std/turn"].append(turn_std_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std/outcome"].append(outcome_std_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std/combined"].append(combined_std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            self._log_completion_samples(prompts, completion_messages, combined_rewards)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
        
    def _prepare_prompt_inputs(self, inputs):
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
            
        return prompt_ids, prompt_mask
    
    def _generate_completions(self, prompts):
        all_prompts = gather_object(prompts)
        if self.accelerator.is_main_process:
            env_result = self.env.generate(
                prompts=all_prompts,
                llm=self.llm,
                sampling_params=self.sampling_params,
            )
            completion_ids = env_result['ids']
            completion_messages = env_result['messages']
            completion_mask = env_result['mask']
            env_assistant_only = env_result.get('assistant_only', [])
            env_tool_logs = env_result.get('tool_logs', [])
            env_raw_datapoint = env_result.get('raw_datapoint', [])
        else:
            completion_ids = [None] * len(all_prompts)
            completion_messages = [None] * len(all_prompts)
            completion_mask = [None] * len(all_prompts)
            env_assistant_only = [None] * len(all_prompts)
            env_tool_logs = [None] * len(all_prompts)
            env_raw_datapoint = [None] * len(all_prompts)

        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        completion_messages = broadcast_object_list(completion_messages, from_process=0)
        completion_mask = broadcast_object_list(completion_mask, from_process=0)
        env_assistant_only = broadcast_object_list(env_assistant_only, from_process=0)
        env_tool_logs = broadcast_object_list(env_tool_logs, from_process=0)
        env_raw_datapoint = broadcast_object_list(env_raw_datapoint, from_process=0)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        completion_ids = completion_ids[process_slice]
        completion_messages = completion_messages[process_slice]
        completion_mask = completion_mask[process_slice]
        env_assistant_only = env_assistant_only[process_slice]
        env_tool_logs = env_tool_logs[process_slice]
        env_raw_datapoint = env_raw_datapoint[process_slice]
        
        device = self.accelerator.device
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)

        completion_mask = [torch.tensor(mask, device=device) for mask in completion_mask]
        completion_mask = pad(completion_mask, padding_value=0)
        
        return completion_ids, completion_messages, completion_mask, env_assistant_only, env_tool_logs, env_raw_datapoint
    
    def _prepare_model_inputs(self, prompt_ids, prompt_mask, completion_ids, completion_mask):
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        return prompt_completion_ids, attention_mask, logits_to_keep
    
    def _compute_logps(self, prompt_completion_ids, attention_mask, logits_to_keep):
        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
                    
        return old_per_token_logps, ref_per_token_logps
    
    def _calculate_rewards(self, prompts, completions, reward_funcs, inputs, env_data=None):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(reward_funcs), device=device)
        
        for i, reward_func in enumerate(reward_funcs):
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
            
            if env_data:
                reward_kwargs.update(env_data)
            
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        return gather(rewards_per_func)
    
    def _log_training_metrics(self, prompts, completions, turn_rewards_per_func, outcome_rewards_per_func, env_raw_datapoint, env_assistant_only, env_tool_logs, inputs):
        import json
        import hashlib
        from datetime import datetime, timezone
        
        if not self.accelerator.is_main_process:
            return
            
        turn_rewards = turn_rewards_per_func.cpu().numpy() if turn_rewards_per_func.numel() > 0 else []
        outcome_rewards = outcome_rewards_per_func.cpu().numpy()
        
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            completion_str = str(completion)[:500] if completion else ""
            conversation_hash = hashlib.md5(f"{prompt[:500]}{completion_str}".encode()).hexdigest()
            
            if conversation_hash in self.logged_conversations:
                continue
            self.logged_conversations.add(conversation_hash)
            
            base_reward = float(outcome_rewards[i, 0]) if outcome_rewards.shape[1] > 0 else 0.0
            tools_reward = float(outcome_rewards[i, 1]) if outcome_rewards.shape[1] > 1 else 0.0
            episodic_naturalness_reward = float(outcome_rewards[i, 2]) if outcome_rewards.shape[1] > 2 else 0.0
            
            turn_based_naturalness_reward = float(turn_rewards[i, 0]) if len(turn_rewards) > 0 and turn_rewards.shape[1] > 0 else 0.0
            
            combined_reward = base_reward + tools_reward + episodic_naturalness_reward + turn_based_naturalness_reward
            
            total_inform = 0
            total_success = 0
            total_intents = 0
            total_success_applicable = 0
            inform_rate = 0.0
            success_rate = 0.0
            individual_informs = []
            individual_successes = []
            intent_details = []
            
            total_book = 0
            book_rate = 0.0
            individual_books = []
            
            base_reward_func = self.outcome_reward_funcs[0] if self.outcome_reward_funcs else None
            if base_reward_func and hasattr(base_reward_func, '_last_detailed_metrics'):
                try:
                    detailed_metrics = base_reward_func._last_detailed_metrics
                    if i < len(detailed_metrics):
                        metrics = detailed_metrics[i]
                        total_inform = metrics.get("total_inform", 0)
                        total_success = metrics.get("total_success", 0)
                        total_intents = metrics.get("total_intents", 0)
                        total_success_applicable = metrics.get("total_success_applicable", 0)
                        inform_rate = metrics.get("inform_rate", 0.0)
                        success_rate = metrics.get("success_rate", 0.0)
                        individual_informs = metrics.get("individual_informs", [])
                        individual_successes = metrics.get("individual_successes", [])
                        intent_details = metrics.get("intent_details", [])
                        
                        total_book = metrics.get("total_book", 0)
                        book_rate = metrics.get("book_rate", 0.0)
                        individual_books = metrics.get("individual_books", [])
                except Exception:
                    pass
            
            tools_breakdown = None
            episodic_naturalness_breakdown = None
            turn_naturalness_breakdown = None
            
            if len(self.outcome_reward_funcs) > 1:
                tools_func = self.outcome_reward_funcs[1]
                if hasattr(tools_func, '_last_reward_breakdowns') and tools_func._last_reward_breakdowns:
                    if i < len(tools_func._last_reward_breakdowns) and tools_func._last_reward_breakdowns[i]:
                        tools_breakdown = tools_func._last_reward_breakdowns[i]
            
            if len(self.outcome_reward_funcs) > 2:
                episodic_naturalness_func = self.outcome_reward_funcs[2]
                if hasattr(episodic_naturalness_func, '_last_reward_breakdowns') and episodic_naturalness_func._last_reward_breakdowns:
                    if i < len(episodic_naturalness_func._last_reward_breakdowns) and episodic_naturalness_func._last_reward_breakdowns[i]:
                        episodic_naturalness_breakdown = episodic_naturalness_func._last_reward_breakdowns[i]
            
            if len(self.turn_reward_funcs) > 0:
                turn_naturalness_func = self.turn_reward_funcs[0]
                if hasattr(turn_naturalness_func, '_last_reward_breakdowns') and turn_naturalness_func._last_reward_breakdowns:
                    if i < len(turn_naturalness_func._last_reward_breakdowns) and turn_naturalness_func._last_reward_breakdowns[i]:
                        turn_naturalness_breakdown = turn_naturalness_func._last_reward_breakdowns[i]
            
            metric_entry = {
                "generation_idx": self.global_generation_counter,
                "reward": combined_reward,
                "base_reward": base_reward,
                "tools_reward": tools_reward,
                "episodic_naturalness_reward": episodic_naturalness_reward,
                "turn_naturalness_reward": turn_based_naturalness_reward,
                "prompt_length": len(prompt) if isinstance(prompt, str) else 0,
                "completion_length": len(str(completion)) if completion else 0,
                "total_inform": total_inform,
                "total_success": total_success,
                "total_intents": total_intents,
                "total_success_applicable": total_success_applicable,
                "inform_rate": inform_rate,
                "success_rate": success_rate,
                "individual_informs": individual_informs,
                "individual_successes": individual_successes,
                "intent_details": intent_details,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            
            if total_book > 0 or book_rate > 0.0 or individual_books:
                metric_entry.update({
                    "total_book": total_book,
                    "book_rate": book_rate,
                    "individual_books": individual_books
                })
            
            if tools_breakdown:
                metric_entry["tools_reward_breakdown"] = tools_breakdown
            
            if episodic_naturalness_breakdown:
                metric_entry["episodic_naturalness_breakdown"] = episodic_naturalness_breakdown
            
            if turn_naturalness_breakdown:
                metric_entry["turn_naturalness_breakdown"] = turn_naturalness_breakdown
            
            with open(self.metrics_log_path, "a") as f:
                f.write(json.dumps(metric_entry) + "\n")
            
            if self.convo_log_path:
                convo_entry = {
                    "generation_idx": self.global_generation_counter,
                    "prompt": prompt,
                    "completion": completion,
                    "reward": combined_reward,
                    "base_reward": base_reward,
                    "tools_reward": tools_reward,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                with open(self.convo_log_path, "a") as f:
                    f.write(json.dumps(convo_entry) + "\n")
            
            if self.actions_log_path and i < len(env_assistant_only):
                assistant_actions = env_assistant_only[i] if env_assistant_only[i] else ""
                if assistant_actions:
                    action_hash = hashlib.md5(assistant_actions.encode()).hexdigest()
                    
                    if action_hash not in self.logged_actions:
                        self.logged_actions.add(action_hash)
                        
                        actions_entry = {
                            "generation_idx": self.global_generation_counter,
                            "assistant_actions": assistant_actions,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        }
                        
                        with open(self.actions_log_path, "a") as f:
                            f.write(json.dumps(actions_entry) + "\n")
            
            self.global_generation_counter += 1
    
    def _compute_normalized_advantages(self, rewards, slice_length=None):
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        
        if hasattr(self.args, 'loss_type') and self.args.loss_type == 'dr_grpo':
            advantages = rewards - mean_grouped_rewards
            std_grouped_rewards = torch.ones_like(mean_grouped_rewards)
        else:
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * slice_length,
            (self.accelerator.process_index + 1) * slice_length,
        )
        advantages = advantages[process_slice]
        
        return mean_grouped_rewards, std_grouped_rewards, advantages
    
    def _log_completion_samples(self, prompts, completions, rewards):
        prompts_to_log = gather_object(prompts)
        completions_to_log = gather_object(completions)
        rewards_to_log = rewards.tolist()

        if self.accelerator.is_main_process:
            if is_rich_available():
                try:
                    if isinstance(prompts_to_log[0][-1], dict):
                        prompt_text = str(prompts_to_log[0][-1]["content"])
                    else:
                        prompt_text = str(prompts_to_log[0][-1])
                except (IndexError, KeyError, TypeError):
                    prompt_text = str(prompts_to_log[0]) if prompts_to_log else "N/A"
                
                print_prompt_completions_sample(
                    [prompt_text],
                    [completions_to_log[0]],
                    [rewards_to_log[0]],
                    self.state.global_step,
                )
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(rewards),
                    "prompt": prompts_to_log,
                    "completion": completions_to_log,
                    "reward": rewards.tolist(),
                }
                df = pd.DataFrame(table)
                wandb.log({"completions": wandb.Table(dataframe=df)})
