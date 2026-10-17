import re
from datetime import datetime
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, AllChem, Descriptors
from tqdm import tqdm, trange
from rdkit import RDLogger

from utils import get_parse, setup_seed, \
    record_operating_parameters, random_mask_smiles, end_epoch_log, \
    set_logger, get_prompt
from load_data import load_test_data
from oracle.drd2_svm import predict_drd2

RDLogger.DisableLog('rdApp.*')

args = get_parse()

logger = set_logger(f'optimize_{args.opti_object}_{args.epochs_test}_{int(10 * args.min_similar)}')
record_operating_parameters(logger, args)


def drd2_online_predict(sl):
    return predict_drd2(sl)[0]


if args.opti_object == "qeddrd2":
    args.opti_object = "qed&drd2"

setup_seed(args.seed)

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
logger.info("loading test end")

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

for epoch_i in trange(0, epochs):
    valid_num = 0

    for step, one_swarm in enumerate(swarms):

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

        for sss in range(len(one_swarm_mask)):
            one_smiles = one_swarm_mask[sss]

            if ('<|mask|>' not in one_smiles) or ('\n' in one_smiles) or (one_smiles.count('<|mask|>') > 1):
                one_swarm_mask[sss] = random_one_swarm_mask[sss]
                continue

            remained_chars = one_smiles.replace('<|mask|>', '')
            is_legal = all(char in allowed_chars for char in remained_chars)
            if not is_legal:
                one_swarm_mask[sss] = random_one_swarm_mask[sss]

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
            if temp_similar[k] < args.min_similar:
                if args.opti_object == "qed&drd2":
                    temp_swarm_fitness.append(-100)
                else:
                    temp_swarm_fitness.append(0)
            else:
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

logger.info("valid rate")
logger.info(valid_num / (test_data_size * args.swarm_num))

qed_pairs = [[init_qed[i], best_qed[i]] for i in range(len(best_qed))]
logp_pairs = [[init_logp[i], best_logp[i]] for i in range(len(best_logp))]
drd2_pairs = [[init_drd2[i], best_drd2[i]] for i in range(len(best_drd2))]
fitness_pairs = [[init_fitness[i], best_fitness[i]] for i in range(len(best_fitness))]
similar_pairs = [[1, best_similar[i]] for i in range(len(best_similar))]

end_epoch_log(args, logger, qed_pairs, logp_pairs, drd2_pairs, fitness_pairs, similar_pairs)

if args.output_file:
    output_file = Path(args.output_file)
else:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    output_file = Path("results") / (
        f"optimized_{args.opti_object}_{args.epochs_test}_{int(10 * args.min_similar)}_{timestamp}.csv"
    )

output_file.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({
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
}).to_csv(output_file, index=False)
logger.info(f"Optimized molecules saved to {output_file}")

logger.info("Testing complete!")
