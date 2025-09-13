This is the codebase to the thesis "Text-To-Agent: Autonomous Generation and Self-Improvement of Task-Specific Agents from Natural-Descriptions".

**Disclaimer:**

Our code uses the codebases from Zeng et al.'s "Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignement" paper (2025) (https://github.com/SiliangZeng/Multi-Turn-RL-Agent) and from Xu et al.'s "Rethinking Task-Oriented Dialogue System: From Complex Modularity to Zero-Shot Autonomous Agent" (2024) (https://github.com/DaDaMrX/AutoTOD) as foundation. We hard-copy and adapt code from these two repositories. In particular:
- The MTRA pre-fixed fine-tuning files contain code from Zeng et al.'s Multi-Turn-RL Agent repo and further parts that are adaptations/adjustments of their code
- The Multi-Turn-RL-Agent folder is a copied version of Zeng et al.'s Multi-Turn-RL Agent repo
- All files pre-fixed with "multiwoz" and "sgd" are entirely copied from Xu et al.'s AutoTOD repo

**Statement on use of AI:**

Both Zeng et al.'s and Xu et al.'s repositories make use of complex data structures, fine-tuning architectures and data logics. To debug the merging of my codebase with the codebase from Zeng et al. and Xu et al., an LLM Assistant in the form of Anthropic's Claude-4-Sonnet was used.

**Organization of this repository:**
- SRC is split into Agent Generator, Finetuning_GRPO, Modular_Learning_Agent and Multi-Turn-RL-Agent
    - Agent Generator contains the code for T2A's first step of generating an agent from a task description
    - Finetuning_GRPO contains the code for GRPO and SFT fine-tuning
    - Modular_Learning_Agent contains the code for the simulation logic and T2A's step of prompt-level improvement
    - Multi-Turn-RL-Agent contains Zeng et al.'s repository
- Scripts contains all the files needed to excute the code on Runpod. All files were run on Runpod with access to GPUs.
- Results contains the results for all experiments in the thesis
- Data contains the SFT data used for training. SGD and MultiWOZ data has to be downloaded from the AutoTOD repository (https://github.com/DaDaMrX/AutoTOD)

**Requirements and environment:**
- A list of requirements can be found in requirements.txt. However, the requirements partly have import dependencies. It is therefore best to use the runpod_venv_setup.sh file to install a virtual environment
- Tokens need to be filled in into the script files and the config files in SRC
- A GPU of at least 48 gigabytes of memory is needed to execute the code. We used an NVIDIA L40S GPU via runpod for our experiments
- All code is executed via the scripts in the Scripts folder. It is strongly recommended to use exactly these scripts on a GPU, as otherwise the execution might fail

**Data:**

As mentioned above, the Data file only contains the SFT data used for training. The SGD and MultiWOZ data has to be downloaded from the AutoTOD repository (https://github.com/DaDaMrX/AutoTOD), following the instructions listed there. The sgd and mwoz named folders then only need to be copied into the Data folder in our repository.

**HuggingFace link to LoRA adapters of trained models (for Llama-3.1-8B-Instruct):**
- Experiment 1: No fine-tuned models
- Experiment 2A:
    - Base Config: ryuotter/t2a_exp2a_sgd_base
    - Tool Config: ryuotter/t2a_exp2a_sgd_tools
    - Naturalness Config: ryuotter/t2a_exp2a_sgd_naturalness
- Experiment 2B:
    - LoRA: ryuotter/t2a_exp2b_sgd_lora
    - Dr. GRPO: ryuotter/t2a_exp2b_sgd_drgrpo
- Experiment 3:
    - SGD:
        - GRPO: ryuotter/t2a_exp3_sgd_grpo
        - SFT: ryuotter/t2a_exp3_sgd_sft
    - MultiWOZ:
        - GRPO: ryuotter/t2a_exp3_mwoz_grpo
        - SFT: ryuotter/t2a_exp3_mwoz_sft