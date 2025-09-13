import os
from pathlib import Path
from .sgd_data_utils import DATA_DIR as SGD_DATA_DIR, load_dialogs as sgd_load_dialogs, load_schemas as sgd_load_schemas, pick_dialog as sgd_pick_dialog 
from .multiwoz_data_utils import DATA_DIR as MULTIWOZ_DATA_DIR, load_data_split as multiwoz_load_data_split
import random
from .improver_logging import get_latest_system_prompt

# Loading MultiWOZ data
def multiwoz_data_loader(split = "train", method = "random", n_points=10, data_dir=MULTIWOZ_DATA_DIR, random_seed=2025):

    data = multiwoz_load_data_split(split, data_dir)
    dialog_ids = list(data.keys())
    
    if method == "random":
        rng = random.Random(random_seed)
        
        if n_points >= len(dialog_ids):
            selected_ids = dialog_ids
        else:
            selected_ids = rng.sample(dialog_ids, n_points)
    elif method == "first":
        selected_ids = dialog_ids[:n_points]
    else:
        raise ValueError("Invalid method")
    
    return {dialog_id: data[dialog_id] for dialog_id in selected_ids}

# Loading SGD data
def sgd_data_loader(domain="SGD", dialog_id="random", data_dir=SGD_DATA_DIR):

    dialogs = sgd_load_dialogs()   
    schemas = sgd_load_schemas()

    individual_dialog = sgd_pick_dialog(dialogs, dialog_id=dialog_id if dialog_id != "random" else "random")
    dialog_id = individual_dialog["dialogue_id"]
    ground_truth_dialog = individual_dialog["turns"]
    datapoint = individual_dialog

    return ground_truth_dialog, dialog_id, datapoint, schemas

# Load SGD eval data
def sgd_eval_data_loader(n=100, seed=42, data_dir=SGD_DATA_DIR):

    dialogs = sgd_load_dialogs(data_dir=data_dir)
    test_ids = [d_id for d_id in dialogs.keys() if d_id.startswith("test_")]

    rng = random.Random(seed)
    selected_ids = sorted(rng.sample(test_ids, n))
    dialogs_list = [dialogs[d_id] for d_id in selected_ids]

    return dialogs_list

# Load SGD train data
def sgd_train_data_loader(n=100, seed=42, splits=["train", "dev"], data_dir=SGD_DATA_DIR):

    dialogs = sgd_load_dialogs(data_dir=data_dir)
    
    available_ids = []
    for split in splits:
        split_ids = [d_id for d_id in dialogs.keys() if d_id.startswith(f"{split}_")]
        available_ids.extend(split_ids)

    rng = random.Random(seed)
    selected_ids = sorted(rng.sample(available_ids, n))
    dialogs_list = [dialogs[d_id] for d_id in selected_ids]

    return dialogs_list

# Load user prompt
def get_user_prompt(benchmark, test_mode=False):

    if test_mode:
        prompt_path = Path(__file__).parent / "prompt_user.txt"
    else:
        prompt_path = Path("prompts") / benchmark / "prompt_user.txt"

    return prompt_path.read_text(encoding="utf-8").strip()

# Load assistant prompt
def get_assistant_prompt(benchmark, domain, prompt_category, iteration="latest", test_mode=False, run_dir=None):

    if run_dir:
        return get_latest_system_prompt(run_dir)

    if test_mode:
        prompt_path = Path(__file__).parent / "prompt_assistant.txt"
        return prompt_path.read_text(encoding="utf-8")