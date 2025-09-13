import os
import yaml
from .utils_accessing import sgd_train_data_loader, multiwoz_data_loader, get_assistant_prompt
from .sgd_data_utils import load_schemas
from .improver_loop import improver_loop
from .improver_logging import setup_run_directory, save_config_snapshot, save_original_prompt, load_yaml_config

# Prompt self-improvement training
def main():

    # Configuration
    config_path = os.path.join(os.path.dirname(__file__), "config_improver.yaml")
    config = load_yaml_config(config_path)
    benchmark = config["benchmark"]
    data_split = config["data_split"]
    domain = config["domain"]
    prompt_category = config["prompt_category"]
    method = config["improver_method"]["method"]
    n_datapoints = config["improver_method"]["n_datapoints"]
    batch_size = config["improver_method"]["batch_size"]
    models = config["llm_models"]
    run_dir = setup_run_directory(config)
    save_config_snapshot(config, run_dir)
    save_original_prompt(run_dir)

    # Loading data
    if benchmark == "SGD":
        train_dialogs = sgd_train_data_loader(n=n_datapoints, seed=42, splits=[data_split])
        schemas = load_schemas()
    elif benchmark == "MultiWOZ":
        data_dict = multiwoz_data_loader(split=data_split, n_points=n_datapoints, random_seed=42)
        train_dialogs = list(data_dict.values())
        schemas = None
    else:
        raise ValueError("False benchmark")
    
    # Improvement loop
    improver_loop(config, train_dialogs, schemas)

if __name__ == "__main__":
    main()