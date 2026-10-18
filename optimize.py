import csv
import json
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, AllChem, Descriptors
from rdkit import rdBase
from tqdm import tqdm, trange
from rdkit import RDLogger

from utils import get_parse, setup_seed, \
    record_operating_parameters, random_mask_smiles, end_epoch_log, \
    set_logger, get_prompt
from load_data import load_test_data
from oracle.drd2_svm import predict_drd2
from checkpointing import (experiment_identity, save_checkpoint, load_checkpoint,
                           restore_checkpoint, mark_complete, sha256, atomic_bytes)
import os

RDLogger.DisableLog('rdApp.*')

args = get_parse()

logger = set_logger(f'optimize_{args.opti_object}_{args.epochs_test}_{int(10 * args.min_similar)}')
record_operating_parameters(logger, args)


def drd2_online_predict(sl):
    return predict_drd2(sl)[0]


if args.opti_object == "qeddrd2":
    args.opti_object = "qed&drd2"

setup_seed(args.seed)

if args.output_file:
    output_file = Path(args.output_file)
else:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    output_file = Path("results") / (
        f"optimized_{args.opti_object}_{args.epochs_test}_"
        f"{int(10 * args.min_similar)}_{timestamp}.csv"
    )
details_dir = output_file.parent / f"{output_file.stem}_details"
if args.resume and not args.output_file:
    raise ValueError("--resume requires --output_file")
if not args.resume and (output_file.exists() or details_dir.exists()):
    raise FileExistsError(
        f"Experiment output already exists: {output_file}"
    )
output_file.parent.mkdir(parents=True, exist_ok=True)
if args.resume and not details_dir.is_dir():
    raise FileNotFoundError(f"No existing run to resume: {details_dir}")
details_dir.mkdir(parents=True, exist_ok=args.resume)
checkpoint_file = details_dir / "checkpoint.pkl"
complete_file = details_dir / "complete.json"
if args.resume and not checkpoint_file.exists():
    raise FileNotFoundError("No resumable checkpoint. Legacy history CSV is not a full population/RNG checkpoint.")
history_file = details_dir / "best_history.csv"
progress_file = details_dir / "progress.csv"
metrics_file = details_dir / "epoch_metrics.csv"
config_file = details_dir / "config.json"
run_log_file = details_dir / "run.log"
file_handler = logging.FileHandler(run_log_file, encoding="utf-8")
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
))
logger.addHandler(file_handler)

logger.info("loading SPA model")
tokenizer = AutoTokenizer.from_pretrained(args.spa_model_path, use_fast=False,
                                          trust_remote_code=True)
tokenizer.padding_side = "left"
model_spa = AutoModelForCausalLM.from_pretrained(args.spa_model_path, device_map="auto",
                                                 torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

logger.info("loading DCR model")
model_dcr = AutoModelForCausalLM.from_pretrained(args.dcr_model_path, device_map="auto",
                                                 torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
p_dcr = get_prompt('dcr_one_mask_prompt_f2')

allowed_chars = ['#', '(', ')', '+', '-', '1', '2', '3', '4', '5', '6', '7', '=', 'B', 'C', 'F', 'H', 'I', 'N', 'O',
                 'P', 'S', '[', ']', 'l', 'r']

logger.info("loading test data")
assert args.opti_object in ["qed", "plogp", "drd2", "qed&drd2"], "Please choose a valid optimization objective"

if args.data_fname_test == "":
    if args.opti_object == "qed":
        args.data_fname_test = 'data/standard_test_data/qed_test.csv'
    if args.opti_object == "plogp":
        args.data_fname_test = 'data/standard_test_data/logp_test.csv'
    if args.opti_object == "drd2":
        args.data_fname_test = 'data/standard_test_data/drd2_test.csv'
    if args.opti_object == "qed&drd2":
        args.data_fname_test = 'data/standard_test_data/drd2qed_test.csv'

test_data, test_data_size = load_test_data(args)
if test_data_size == 0 or args.epochs_test < 1 or args.swarm_num < 1:
    raise ValueError("Data, epochs_test and swarm_num must be nonempty/positive")
logger.info("loading test end")


def get_git_info(*cmd):
    try:
        return subprocess.check_output(
            ["git", *cmd],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


config = {
    "start_time": datetime.now().isoformat(),
    "git_commit": get_git_info("rev-parse", "HEAD"),
    "git_status": get_git_info("status", "--porcelain"),
    "parameters": vars(args).copy(),
    "test_data_size": test_data_size,
    "mask_check": "author_original",
    "python_version": sys.version,
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "rdkit_version": rdBase.rdkitVersion,
    "similarity_rule": "Morgan count radius=2, Tanimoto > min_similar (author code)",
}
identity = experiment_identity(args)
resume_state = load_checkpoint(checkpoint_file, identity) if args.resume else None
if args.resume and complete_file.exists():
    complete = json.loads(complete_file.read_text(encoding="utf-8"))
    if complete['identity'] != identity or complete['csv_sha256'] != sha256(output_file):
        raise ValueError("Completed result verification failed")
    logger.info("Verified completed run; no optimization repeated.")
    sys.exit(0)
if not args.resume:
    config['identity'] = identity
    config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

epochs = args.epochs_test
swarm_num = args.swarm_num


def safe_decode(g_id, tknz):
    tknz_list = []
    for idx, ids in enumerate(g_id):
        decoded_tokens = []
        for token_idx, token_id in enumerate(ids):
            try:
                decoded_token = tknz.decode([token_id], skip_special_tokens=True)
                decoded_tokens.append(decoded_token)
            except Exception as e:
                logger.info(f"Token decoding error at index {token_idx} in generated_ids[{idx}]: {e}")
                continue
        tknz_list.append("".join(decoded_tokens))
    return tknz_list


swarms = []
swarms_qed = []
swarms_logp = []
swarms_drd2 = []

init_fitness = []
init_qed = []
init_logp = []
init_drd2 = []

best_fitness = []
best_qed = []
best_logp = []
best_drd2 = []
best_similar = []
best_smiles = []


def make_results_df():
    return pd.DataFrame({
        "input_smiles": test_data.smiles,
        "optimized_smiles": best_smiles,
        "initial_fitness": init_fitness,
        "optimized_fitness": best_fitness,
        "initial_qed": init_qed,
        "optimized_qed": best_qed,
        "initial_logp": init_logp,
        "optimized_logp": best_logp,
        "initial_drd2": init_drd2,
        "optimized_drd2": best_drd2,
        "similarity": best_similar,
    })


def atomic_save_csv(df, path):
    atomic_bytes(path, df.to_csv(index=False).encode("utf-8"))


def append_csv_row(path, row):
    is_new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


for step, one_sm in tqdm(enumerate(test_data.smiles), total=len(test_data.smiles), desc="init..."):
    swarms.append([one_sm] * swarm_num)

    molecule = Chem.MolFromSmiles(one_sm)

    one_sm_qed = QED.qed(molecule)
    one_sm_logp = Descriptors.MolLogP(molecule)
    one_sm_drd2 = drd2_online_predict([one_sm])

    init_qed.append(one_sm_qed)
    init_logp.append(one_sm_logp)
    init_drd2.append(one_sm_drd2)

    if args.opti_object == "qed":
        init_fitness.append(one_sm_qed)
        best_fitness.append(one_sm_qed)
    elif args.opti_object == "plogp":
        init_fitness.append(one_sm_logp)
        best_fitness.append(one_sm_logp)
    elif args.opti_object == "drd2":
        init_fitness.append(one_sm_drd2)
        best_fitness.append(one_sm_drd2)
    else:
        initial_multi_fitness = min(one_sm_qed, 0.6) + min(one_sm_drd2, 0.5)
        init_fitness.append(initial_multi_fitness)
        best_fitness.append(initial_multi_fitness)

    swarms_qed.append([one_sm_qed] * swarm_num)
    swarms_logp.append([one_sm_logp] * swarm_num)
    swarms_drd2.append([one_sm_drd2] * swarm_num)

    best_qed.append(one_sm_qed)
    best_logp.append(one_sm_logp)
    best_drd2.append(one_sm_drd2)
    best_similar.append(1)
    best_smiles.append(one_sm)

start_epoch, start_step = 0, 0
journals = (history_file, metrics_file)
if resume_state:
    restore_checkpoint(resume_state, globals())
    start_epoch, start_step = resume_state['epoch'], resume_state['step']
    logger.info(f"Resuming epoch index={start_epoch}, next molecule index={start_step}")
else:
    initial_df = make_results_df()
    initial_df.insert(0, "epoch", 0)
    initial_df.insert(1, "molecule_id", range(len(initial_df)))
    atomic_save_csv(initial_df, progress_file)
    save_checkpoint(checkpoint_file, globals(), identity, 0, 0, (0, 0, 0), journals)
    logger.info("Initial molecule results and checkpoint saved")

for epoch_i in trange(start_epoch, epochs):
    valid_num, mask_total_count, mask_fallback_count = (
        resume_state['counters'] if resume_state and epoch_i == start_epoch else (0, 0, 0)
    )

    for step, one_swarm in enumerate(swarms):
        if epoch_i == start_epoch and step < start_step:
            continue

        one_swarm_qed = swarms_qed[step]
        one_swarm_logp = swarms_logp[step]
        one_swarm_drd2 = swarms_drd2[step]

        random_one_swarm_mask, _ = random_mask_smiles(one_swarm, args)

        input_ids_list = []
        for pre in one_swarm:
            molecule_pre = Chem.MolFromSmiles(pre)
            qed_pre = QED.qed(molecule_pre)
            logp_pre = Descriptors.MolLogP(molecule_pre)
            drd2_pre = drd2_online_predict([pre])

            proo = None
            if args.opti_object == "qed":
                proo = round(qed_pre, 2)
            elif args.opti_object == "plogp":
                proo = round(logp_pre, 2)
            elif args.opti_object == "drd2":
                proo = round(drd2_pre, 2)
            else:
                proo = (round(qed_pre, 2), round(drd2_pre, 2))

            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": pre + '<|>' + args.opti_object + '=' + str(proo) + '<|>'}
            ]
            input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids_list.append(input_ids)

        model_inputs = tokenizer(input_ids_list, return_tensors="pt", padding=True, truncation=True).to('cuda')

        generated_ids = model_spa.generate(model_inputs.input_ids,attention_mask=model_inputs.attention_mask,
                                           max_new_tokens=args.spa_max_length,
                                           do_sample=True, top_p=0.97)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        one_swarm_mask = safe_decode(generated_ids, tokenizer)

        one_swarm_mask = [one_smiles.replace('Q', '&') for one_smiles in one_swarm_mask]

        mask_total_count += len(one_swarm_mask)
        for sss in range(len(one_swarm_mask)):
            one_smiles = one_swarm_mask[sss]

            if ('<|mask|>' not in one_smiles) or ('\n' in one_smiles) or (one_smiles.count('<|mask|>') > 1):
                one_swarm_mask[sss] = random_one_swarm_mask[sss]
                mask_fallback_count += 1
                continue

            remained_chars = one_smiles.replace('<|mask|>', '')
            is_legal = all(char in allowed_chars for char in remained_chars)
            if not is_legal:
                one_swarm_mask[sss] = random_one_swarm_mask[sss]
                mask_fallback_count += 1

        new_one_swarm = []
        new_one_swarm_qed = []
        new_one_swarm_logp = []
        new_one_swarm_drd2 = []

        one_swarm_mask = [one_smiles.replace('&', 'Q') for one_smiles in one_swarm_mask]

        if args.use_one_mask:
            def replace_continuous_q(s):
                if 'Q' in s:
                    return re.sub(r'Q+', '<|mask|>', s)
                return s

            one_swarm_mask = [replace_continuous_q(one_smiles) for one_smiles in one_swarm_mask]

        mid_list = []
        for mid_Q in one_swarm_mask:
            messages = [
                {"role": "system", "content": p_dcr},
                {"role": "user", "content": mid_Q + '<|>' + args.opti_object + '<|>'}
            ]
            input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')

            generated_ids = model_dcr.generate(model_inputs.input_ids,
                                               attention_mask=model_inputs.attention_mask,
                                               max_new_tokens=args.dcr_max_length)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            mid_list.append(response)

        mid_list = [one_mask.replace('<|mask|>', mid) for one_mask, mid in zip(one_swarm_mask, mid_list)]

        for mid_i in range(len(mid_list)):
            try:
                mid = mid_list[mid_i]
                molecule = Chem.MolFromSmiles(mid)

                qed = QED.qed(molecule)
                logp = Descriptors.MolLogP(molecule)
                drd2 = drd2_online_predict([mid])

                new_one_swarm_qed.append(qed)
                new_one_swarm_logp.append(logp)
                new_one_swarm_drd2.append(drd2)
                new_one_swarm.append(mid)
                valid_num = valid_num + 1

            except Exception:
                new_one_swarm.append(one_swarm[mid_i])
                new_one_swarm_qed.append(one_swarm_qed[mid_i])
                new_one_swarm_logp.append(one_swarm_logp[mid_i])
                new_one_swarm_drd2.append(one_swarm_drd2[mid_i])

        no_repeat_swarm = []
        no_repeat_swarm_qed = []
        no_repeat_swarm_logp = []
        no_repeat_swarm_drd2 = []

        for qq in range(len(new_one_swarm)):
            if new_one_swarm[qq] not in one_swarm:
                no_repeat_swarm.append(new_one_swarm[qq])
                no_repeat_swarm_qed.append(new_one_swarm_qed[qq])
                no_repeat_swarm_logp.append(new_one_swarm_logp[qq])
                no_repeat_swarm_drd2.append(new_one_swarm_drd2[qq])

        if args.allow_individuals_repeat:
            temp_swarm = one_swarm + new_one_swarm
            temp_swarm_qed = one_swarm_qed + new_one_swarm_qed
            temp_swarm_logp = one_swarm_logp + new_one_swarm_logp
            temp_swarm_drd2 = one_swarm_drd2 + new_one_swarm_drd2
        else:
            temp_swarm = one_swarm + no_repeat_swarm
            temp_swarm_qed = one_swarm_qed + no_repeat_swarm_qed
            temp_swarm_logp = one_swarm_logp + no_repeat_swarm_logp
            temp_swarm_drd2 = one_swarm_drd2 + no_repeat_swarm_drd2

        temp_similar = []
        lead_smiles = test_data.smiles[step]
        for sim_i in temp_swarm:
            x_1 = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(lead_smiles), 2)
            try:
                x_2 = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(sim_i), 2)
                similarity = DataStructs.TanimotoSimilarity(x_1, x_2)
                temp_similar.append(similarity)
            except Exception:
                temp_similar.append(0)

        temp_swarm_fitness = []
        for k in range(len(temp_swarm_qed)):
            if temp_similar[k] <= args.min_similar:
                temp_swarm_fitness.append(float("-inf"))
                continue

            f = 0
            if args.opti_object == "qed":
                f = temp_swarm_qed[k]
            elif args.opti_object == "plogp":
                f = temp_swarm_logp[k]
            elif args.opti_object == "drd2":
                f = temp_swarm_drd2[k]
            elif args.opti_object == "qed&drd2":
                f = min(temp_swarm_qed[k], 0.6) + min(temp_swarm_drd2[k], 0.5)
            temp_swarm_fitness.append(f)

        molecular_data = list(zip(temp_swarm, temp_swarm_fitness, temp_swarm_qed, temp_similar, temp_swarm_logp,
                                  temp_swarm_drd2))

        sorted_data = sorted(molecular_data, key=lambda x: x[1], reverse=True)

        top_5_swarm = [x[0] for x in sorted_data[:swarm_num]]
        top_5_swarm_fitness = [x[1] for x in sorted_data[:swarm_num]]
        top_5_swarm_qed = [x[2] for x in sorted_data[:swarm_num]]
        top_5_swarm_logp = [x[4] for x in sorted_data[:swarm_num]]
        top_5_swarm_drd2 = [x[5] for x in sorted_data[:swarm_num]]
        top_5_swarm_similar = [x[3] for x in sorted_data[:swarm_num]]

        max_swarm_fitness = max(top_5_swarm_fitness)
        if max_swarm_fitness > best_fitness[step]:
            max_index = top_5_swarm_fitness.index(max_swarm_fitness)
            best_fitness[step] = max_swarm_fitness
            best_qed[step] = top_5_swarm_qed[max_index]
            best_logp[step] = top_5_swarm_logp[max_index]
            best_drd2[step] = top_5_swarm_drd2[max_index]
            best_similar[step] = top_5_swarm_similar[max_index]
            best_smiles[step] = top_5_swarm[max_index]

        swarms[step] = top_5_swarm
        swarms_qed[step] = top_5_swarm_qed
        swarms_logp[step] = top_5_swarm_logp
        swarms_drd2[step] = top_5_swarm_drd2

        history_row = {
            "epoch": epoch_i + 1,
            "molecule_id": step,
            "input_smiles": test_data.smiles[step],
            "optimized_smiles": best_smiles[step],
            "initial_fitness": init_fitness[step],
            "optimized_fitness": best_fitness[step],
            "initial_qed": init_qed[step],
            "optimized_qed": best_qed[step],
            "initial_logp": init_logp[step],
            "optimized_logp": best_logp[step],
            "initial_drd2": init_drd2[step],
            "optimized_drd2": best_drd2[step],
            "similarity": best_similar[step],
        }
        append_csv_row(history_file, history_row)
        save_checkpoint(checkpoint_file, globals(), identity, epoch_i, step + 1,
                        (valid_num, mask_total_count, mask_fallback_count), journals)
        if (step + 1) % 50 == 0:
            logger.info(f"Epoch {epoch_i + 1}/{epochs}, molecule {step + 1}/{test_data_size}: checkpoint saved")

    progress_df = make_results_df()
    progress_df.insert(0, "epoch", epoch_i + 1)
    progress_df.insert(1, "molecule_id", range(len(progress_df)))
    atomic_save_csv(progress_df, progress_file)

    n = test_data_size
    metrics_row = {
        "epoch": epoch_i + 1,
        "num_molecules": n,
        "valid_generation_rate":
            valid_num / (n * swarm_num),
        "mask_fallback_count": mask_fallback_count,
        "mask_fallback_rate":
            mask_fallback_count / mask_total_count
            if mask_total_count else 0,
        "mean_qed": sum(best_qed) / n,
        "mean_logp": sum(best_logp) / n,
        "mean_drd2": sum(best_drd2) / n,
        "mean_similarity": sum(best_similar) / n,
        "mean_qed_improvement":
            sum(a - b for a, b in zip(best_qed, init_qed)) / n,
        "mean_logp_improvement":
            sum(a - b for a, b in zip(best_logp, init_logp)) / n,
        "mean_drd2_improvement":
            sum(a - b for a, b in zip(best_drd2, init_drd2)) / n,
        "qed_ge_0_9_percent":
            100 * sum(x >= 0.9 for x in best_qed) / n,
        "qed_delta_ge_0_1_percent":
            100 * sum(
                a - b >= 0.1
                for a, b in zip(best_qed, init_qed)
            ) / n,
        "drd2_ge_0_5_percent":
            100 * sum(x >= 0.5 for x in best_drd2) / n,
        "drd2_delta_ge_0_2_percent":
            100 * sum(
                a - b >= 0.2
                for a, b in zip(best_drd2, init_drd2)
            ) / n,
        "mo_success_percent":
            100 * sum(
                q > 0.6 and d > 0.5
                for q, d in zip(best_qed, best_drd2)
            ) / n,
    }
    append_csv_row(metrics_file, metrics_row)
    save_checkpoint(checkpoint_file, globals(), identity, epoch_i + 1, 0,
                    (0, 0, 0), journals)
    logger.info(
        f"Epoch {epoch_i + 1}/{epochs}: "
        f"valid_rate={metrics_row['valid_generation_rate']:.4f}, "
        f"mask_fallback_rate={metrics_row['mask_fallback_rate']:.4f}"
    )

qed_pairs = [[init_qed[i], best_qed[i]] for i in range(len(best_qed))]
logp_pairs = [[init_logp[i], best_logp[i]] for i in range(len(best_logp))]
drd2_pairs = [[init_drd2[i], best_drd2[i]] for i in range(len(best_drd2))]
fitness_pairs = [[init_fitness[i], best_fitness[i]] for i in range(len(best_fitness))]
similar_pairs = [[1, best_similar[i]] for i in range(len(best_similar))]

atomic_save_csv(make_results_df(), output_file)
mark_complete(complete_file, output_file, identity, test_data_size)
logger.info(f"Optimized molecules saved to {output_file}")
end_epoch_log(args, logger, qed_pairs, logp_pairs, drd2_pairs, fitness_pairs, similar_pairs)

logger.info("Testing complete!")
