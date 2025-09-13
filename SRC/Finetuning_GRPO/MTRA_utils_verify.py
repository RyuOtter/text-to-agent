from __future__ import annotations
import os
import glob
import json
from safetensors import safe_open
import torch

# Check if LoRA adapter file contains non-zero weights
def _check_file_nonzero(path):
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            if (t == 0).sum().item() == t.numel():
                return False
    return True

# Analyze LoRA adapter weights
def _analyze_adapter_weights(path):
    
    stats = {
        "total_params": 0,
        "zero_params": 0,
        "mean_magnitude": 0.0,
        "max_magnitude": 0.0,
        "layer_stats": {},
        "rank_info": {}
    }
    
    all_magnitudes = []
    
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            
            total_params = tensor.numel()
            zero_params = (tensor == 0).sum().item()
            mean_mag = tensor.abs().mean().item()
            max_mag = tensor.abs().max().item()
            
            stats["total_params"] += total_params
            stats["zero_params"] += zero_params
            all_magnitudes.extend(tensor.flatten().abs().tolist())
            
            stats["layer_stats"][key] = {
                "shape": list(tensor.shape),
                "total_params": total_params,
                "zero_params": zero_params,
                "zero_ratio": zero_params / total_params,
                "mean_magnitude": mean_mag,
                "max_magnitude": max_mag
            }
            
            if "lora_A" in key or "lora_B" in key:
                try:
                    if len(tensor.shape) == 2:
                        u, s, v = torch.svd(tensor)
                        effective_rank = (s > 1e-6).sum().item()
                        stats["rank_info"][key] = {
                            "nominal_rank": min(tensor.shape),
                            "effective_rank": effective_rank,
                            "rank_utilization": effective_rank / min(tensor.shape)
                        }
                except Exception:
                    pass
    
    if stats["total_params"] > 0:
        stats["overall_zero_ratio"] = stats["zero_params"] / stats["total_params"]
        stats["mean_magnitude"] = sum(all_magnitudes) / len(all_magnitudes) if all_magnitudes else 0.0
        stats["max_magnitude"] = max(all_magnitudes) if all_magnitudes else 0.0
    
    return stats

# Verification of LoRA weights (partly redundant function to the other, left over from older code)
def enhanced_verify_lora_adapters(output_dir):
    search_dirs = [output_dir]
    
    results_base = os.path.dirname(os.path.dirname(output_dir)) if output_dir else "/workspace/thesis/Results"
    
    if os.path.exists(results_base):
        all_results = [d for d in os.listdir(results_base) if os.path.isdir(os.path.join(results_base, d))]
        all_results.sort(reverse=True) 
        
        for result_dir in all_results[:3]: 
            checkpoints_dir = os.path.join(results_base, result_dir, "checkpoints")
            if os.path.exists(checkpoints_dir):
                checkpoint_dirs = [d for d in os.listdir(checkpoints_dir) 
                                 if os.path.isdir(os.path.join(checkpoints_dir, d)) and d.startswith("checkpoint-")]
                if checkpoint_dirs:
                    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))
                    search_dirs.append(os.path.join(checkpoints_dir, latest_checkpoint))
                    break
    
    candidates = []
    seen_files = set()
    for search_dir in search_dirs:
        adapter_patterns = [
            os.path.join(search_dir, "adapter_model.safetensors"),
            os.path.join(search_dir, "*adapter*.safetensors"),
        ]
        for p in adapter_patterns:
            for file_path in glob.glob(p, recursive=False):
                if file_path not in seen_files:
                    candidates.append(file_path)
                    seen_files.add(file_path)

    results = {
        "success": True,
        "files_checked": len(candidates),
        "file_results": {},
        "overall_stats": {
            "total_params": 0,
            "zero_params": 0,
            "effective_rank_avg": 0.0
        }
    }
    
    total_effective_ranks = []
    
    for fpath in candidates:
        rel_path = os.path.relpath(fpath, output_dir)
        
        try:
            basic_ok = _check_file_nonzero(fpath)
            stats = _analyze_adapter_weights(fpath)
            
            results["overall_stats"]["total_params"] += stats["total_params"]
            results["overall_stats"]["zero_params"] += stats["zero_params"]
            
            for rank_info in stats["rank_info"].values():
                if "rank_utilization" in rank_info:
                    total_effective_ranks.append(rank_info["rank_utilization"])
            
            file_result = {
                "basic_check": basic_ok,
                "stats": stats,
                "assessment": "GOOD" if basic_ok and stats["overall_zero_ratio"] < 0.9 else "POOR"
            }
            
            results["file_results"][rel_path] = file_result
            
            if not basic_ok:
                results["success"] = False
                
        except Exception as e:
            results["success"] = False
            results["file_results"][rel_path] = {"basic_check": False, "error": str(e)}

    if results["overall_stats"]["total_params"] > 0:
        overall_zero_ratio = results["overall_stats"]["zero_params"] / results["overall_stats"]["total_params"]
        results["overall_stats"]["zero_ratio"] = overall_zero_ratio
        
    if total_effective_ranks:
        results["overall_stats"]["effective_rank_avg"] = sum(total_effective_ranks) / len(total_effective_ranks)
    
    print(f"Passed" if results["success"] else "Failed")
    
    return results

# Verify adapters not zero (partly redundant functions to the others, left over from older code)
def verify_lora_adapters(output_dir):

    search_dirs = [output_dir]
    results_base = os.path.dirname(os.path.dirname(output_dir)) if output_dir else "/workspace/thesis/Results"
    
    if os.path.exists(results_base):
        all_results = [d for d in os.listdir(results_base) if os.path.isdir(os.path.join(results_base, d))]
        all_results.sort(reverse=True)
        
        for result_dir in all_results[:3]:
            checkpoints_dir = os.path.join(results_base, result_dir, "checkpoints")
            if os.path.exists(checkpoints_dir):
                checkpoint_dirs = [d for d in os.listdir(checkpoints_dir) 
                                 if os.path.isdir(os.path.join(checkpoints_dir, d)) and d.startswith("checkpoint-")]
                if checkpoint_dirs:
                    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))
                    search_dirs.append(os.path.join(checkpoints_dir, latest_checkpoint))
                    break
    
    candidates = []
    seen_files = set()
    for search_dir in search_dirs:
        adapter_patterns = [
            os.path.join(search_dir, "adapter_model.safetensors"),
            os.path.join(search_dir, "*adapter*.safetensors"),
        ]
        for p in adapter_patterns:
            for file_path in glob.glob(p, recursive=False):
                if file_path not in seen_files:
                    candidates.append(file_path)
                    seen_files.add(file_path)

    ok_all = True
    for fpath in candidates:
        ok = _check_file_nonzero(fpath)
        rel_path = fpath
        for search_dir in search_dirs:
            if fpath.startswith(search_dir):
                rel_path = os.path.relpath(fpath, search_dir)
                break
        ok_all = ok_all and ok
    return ok_all