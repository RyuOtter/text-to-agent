from datetime import datetime
from pathlib import Path
import yaml
import os

# Load config
def load_yaml_config(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data

# Create new directory for results
def setup_run_directory(config):
    base_dir = config["logging"]["base_dir"]
    timestamp_format = config["logging"]["timestamp_format"]
    benchmark = config["benchmark"]
    method = config["improver_method"]["method"]
    
    timestamp = datetime.now().strftime(timestamp_format)
    run_id = f"{method}_{timestamp}"
    
    run_dir = Path(base_dir) / benchmark / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    config["run_info"]["run_id"] = run_id
    config["run_info"]["run_dir"] = str(run_dir)
    
    return str(run_dir)

# Save config
def save_config_snapshot(config, run_dir):
    config_path = Path(run_dir) / "config_snapshot.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

# Save initial starting assistant prompt
def save_original_prompt(run_dir):
    original_prompt_path = Path(__file__).parent / "prompt_assistant_min.txt"
    
    with open(original_prompt_path, "r", encoding="utf-8") as f:
        original_prompt = f.read()
    
    iteration_path = Path(run_dir) / "prompt_iteration_000.txt"
    with open(iteration_path, "w", encoding="utf-8") as f:
        f.write(original_prompt)

# Load the latest system prompt
def get_latest_system_prompt(run_dir):
    run_path = Path(run_dir)
    iteration_files = sorted(run_path.glob("prompt_iteration_*.txt"))
    
    latest_file = iteration_files[-1]
    
    with open(latest_file, "r", encoding="utf-8") as f:
        return f.read()