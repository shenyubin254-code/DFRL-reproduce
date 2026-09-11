# Model Weights

The trained weights are **not** distributed through this repository because of
their size. Download the archive from the link announced with the paper (see
the main README) and extract it into the repository root, so that the
following paths resolve:

| Path                                     | Model                                   |
| ---------------------------------------- | --------------------------------------- |
| `mask_grpo_model/2025120822_stage_4/checkpoint-10000/` | SPA (mask / localization, GRPO-trained) |
| `env_sft_model/2025061014/checkpoint-300000/`          | DCR (env / completion, SFT-trained)     |
| `models/drd2/clf_py36.pkl`               | DRD2 oracle (SVM)                       |

Equivalently you may keep the checkpoints anywhere and pass the paths to
`optimize.py` via `--spa_model_path` / `--dcr_model_path`, and set
`DRD2_MODEL_PATH` (see `oracle/drd2_svm.py`) for the oracle.

Each LLM directory must be a standard HuggingFace checkpoint layout
(`config.json`, `model.safetensors`, tokenizer files). Both SPA and DCR are
fine-tuned from Qwen2.5-0.5B-Instruct, so their checkpoints contain their own
tokenizer files.

The DRD2 oracle (`clf_py36.pkl`) is an sklearn SVC trained on DRD2 activity
data from ExCAPE (obtained via [TDC](https://tdcommons.ai/)); its inference
code is `oracle/drd2_svm.py` and has been verified to reproduce the DRD2
values of the paper exactly. No oracle weights are committed to the repo.
