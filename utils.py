import argparse
import ast
import logging
import os
import random
import re
import time

import requests
import torch
import numpy as np

ORACLE_HOST = os.environ.get("ORACLE_HOST", "http://localhost:5002")


def drd2_predict_oracle(sl):
    response = requests.post(ORACLE_HOST + "/predict", json={"SMILES_LIST": sl})
    result_list = ast.literal_eval(response.json()["result"])
    if isinstance(result_list, float):
        return result_list
    if len(result_list) == 1:
        return float(result_list[0])
    return [float(i) for i in result_list]


def get_prompt(prompt_name):
    prompt_dict = {
        "dcr_prompt": "输入一个不完整的分子式，不完整的部分用Q替代，你将其修正为完整的分子式。\n不需要给出具体过程，直接给出答案。\nexample1：\ninput：COC1=C(N)C=C2QQQQQQQC1O<|>plogp<|>\noutput：COC1=C(N)C=C2SC=CC2=C1O\nexample2：\ninput2：CCCNC1CCOCQQQQQQCCC(C)C<|>qed&drd2<|>\noutput：CCCNC1CCOCC1N(C)CCC(C)C",
        "dcr_one_mask_prompt_f2": "输入一个不完整的分子式，不完整的部分用<|mask|>替代，你输出被mask的部分，使分子完整且合法。\n不需要给出具体过程，直接给出答案。\nexample1：\ninput：COC1=C(N)C=C2<|mask|>C1O<|>plogp<|>\noutput：SC=CC2=\nexample2：\ninput：CCCNC1CCOC<|mask|>CCC(C)C<|>qed&drd2<|>\noutput：C1N(C)",
    }
    return prompt_dict[prompt_name]


def check_q_replacement_3(target: str, completion_content: str) -> bool:
    len_target = len(target)
    len_completion_content = len(completion_content)
    if len_target != len_completion_content:
        return False
    q_count = completion_content.count("Q")
    if q_count == 0:
        return False
    for i in range(len_target - q_count + 1):
        transformed_target = target[:i] + "Q" * q_count + target[i + q_count:]
        if transformed_target == completion_content:
            return True
    return False


def end_epoch_log(args, logger, qed_pairs, logp_pairs, drd2_pairs, fitness_pairs, similar_pairs):
    if args.opti_object == "qed":
        num = 0
        aaa = 0
        bbb = 0
        for i in qed_pairs:
            num = num + 1
            if i[-1] > 0.9:
                aaa = aaa + 1
            if i[-1] - i[0] > 0.1:
                bbb = bbb + 1
        logger.info("qed > 0.9 ratio")
        logger.info(aaa / num)
        logger.info("qed improvement > 0.1 ratio")
        logger.info(bbb / num)

        count_better = 0
        better_average = []
        if args.epochs_test == 1:
            for i in qed_pairs:
                if i[-1] > i[0]:
                    count_better = count_better + 1
                    better_average.append(i[-1] - i[0])
            logger.info("improvement ratio")
            logger.info(count_better / num)
            logger.info("average improvement")
            logger.info(sum(better_average) / len(better_average))

    if args.opti_object in ("logp", "plogp"):
        before = []
        after = []
        for i in logp_pairs:
            before.append(i[0])
            after.append(i[-1])
        logger.info("avg logp before")
        logger.info(sum(before) / len(before))
        logger.info("avg logp after")
        logger.info(sum(after) / len(after))

    if args.opti_object == "drd2":
        before = []
        after = []
        for i in drd2_pairs:
            before.append(i[0])
            after.append(i[-1])
        logger.info("avg drd2 before")
        logger.info(sum(before) / len(before))
        logger.info("avg drd2 after")
        logger.info(sum(after) / len(after))

        num_d = 0
        aaa_q = 0
        aaa_d = 0
        for i in drd2_pairs:
            num_d = num_d + 1
            if i[-1] > 0.5:
                aaa_q = aaa_q + 1
            if i[-1] - i[0] > 0.2:
                aaa_d = aaa_d + 1
        logger.info("drd2 > 0.5 ratio")
        logger.info(aaa_q / num_d)
        logger.info("drd2 improvement > 0.2 ratio")
        logger.info(aaa_d / num_d)

    if args.opti_object == "qed&drd2":
        num_d = 0
        aaa_q = 0
        aaa_d = 0
        aaa_q_d = 0
        for i in qed_pairs:
            num_d = num_d + 1
            if i[-1] > 0.6:
                aaa_q = aaa_q + 1
        for j in drd2_pairs:
            if j[-1] > 0.5:
                aaa_d = aaa_d + 1
        for k in range(len(qed_pairs)):
            if qed_pairs[k][-1] > 0.6 and drd2_pairs[k][-1] > 0.5:
                aaa_q_d = aaa_q_d + 1
        logger.info("qed > 0.6 ratio")
        logger.info(aaa_q / num_d)
        logger.info("drd2 > 0.5 ratio")
        logger.info(aaa_d / num_d)
        logger.info("qed > 0.6 and drd2 > 0.5 ratio")
        logger.info(aaa_q_d / num_d)

    final_average_fitness = [row[-1] for row in fitness_pairs]
    logger.info("final average fitness")
    logger.info(sum(final_average_fitness) / len(final_average_fitness))

    final_average_qed = [row[-1] for row in qed_pairs]
    logger.info("final average qed")
    logger.info(sum(final_average_qed) / len(final_average_qed))

    final_average_logp = [row[-1] for row in logp_pairs]
    logger.info("final average logp")
    logger.info(sum(final_average_logp) / len(final_average_logp))

    final_average_drd2 = [row[-1] for row in drd2_pairs]
    logger.info("final average drd2")
    logger.info(sum(final_average_drd2) / len(final_average_drd2))

    final_average_similar = [row[-1] for row in similar_pairs]
    logger.info("final average similar")
    logger.info(sum(final_average_similar) / len(final_average_similar))


def random_mask_smiles(swarm, args):
    new_swarm = []
    replaced_chars = []

    for i in range(args.swarm_num):
        string = swarm[i]
        length = len(string)

        start = random.randint(0, length - 1)
        end = random.randint(start + 1, min(start + args.max_mask_len, length))

        replaced = string[start:end]
        replaced_chars.append(replaced)

        Q_number = random.randint(1, args.max_mask_len)
        replaced_string = string[:start] + "Q" * Q_number + string[end:]

        if args.only_one_mask:
            replaced_string = re.sub(r"Q+", "&", replaced_string)
        else:
            replaced_string = replaced_string.replace("Q", "&")

        new_swarm.append(replaced_string)

    return new_swarm, replaced_chars


def set_logger(name=""):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rq = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))
    random_number = random.randint(1000, 9999)
    log_name = "./log/" + name + "_" + rq + "_" + str(random_number) + ".log"
    os.makedirs("./log", exist_ok=True)
    fh = logging.FileHandler(log_name, mode="w")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def record_operating_parameters(logger, args):
    logger.info("------------------------------for test------------------------------")
    logger.info("spa_model_path")
    logger.info(args.spa_model_path)
    logger.info("dcr_model_path")
    logger.info(args.dcr_model_path)
    logger.info("epochs_test")
    logger.info(args.epochs_test)
    logger.info("min_similar")
    logger.info(args.min_similar)
    logger.info("swarm_num")
    logger.info(args.swarm_num)
    logger.info("opti_object")
    logger.info(args.opti_object)
    logger.info("use_one_mask")
    logger.info(args.use_one_mask)
    logger.info("spa_max_length")
    logger.info(args.spa_max_length)
    logger.info("dcr_max_length")
    logger.info(args.dcr_max_length)
    logger.info("max_mask_len")
    logger.info(args.max_mask_len)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--spa_model_path", type=str, default="models/spa",
                        help="Path to the SPA (mask/localization) model checkpoint.")
    parser.add_argument("--dcr_model_path", type=str, default="models/dcr",
                        help="Path to the DCR (env/completion) model checkpoint.")
    parser.add_argument("--epochs_test", type=int, default=20)
    parser.add_argument("--min_similar", type=float, default=0.4,
                        help="Tanimoto similarity threshold (delta) to the lead molecule.")
    parser.add_argument("--swarm_num", type=int, default=20)
    parser.add_argument("--batch_size_test", type=int, default=20)
    parser.add_argument("--opti_object", type=str, default="qed",
                        help="Optimization objective: qed / plogp / drd2 / qed&drd2 "
                             "(qeddrd2 is accepted as an alias of qed&drd2).")
    parser.add_argument("--use_one_mask", action="store_true",
                        help="Use a single <|mask|> token for the masked span.")
    parser.add_argument("--only_one_mask", action="store_true")
    parser.add_argument("--max_mask_len", type=int, default=5)
    parser.add_argument("--spa_max_length", type=int, default=512,
                        help="Max new tokens generated by the SPA (mask model).")
    parser.add_argument("--dcr_max_length", type=int, default=512,
                        help="Max new tokens generated by the DCR (env model).")
    parser.add_argument("--allow_individuals_repeat", action="store_true")

    parser.add_argument("--data_fname_test", type=str, default="")
    parser.add_argument("--output_file", type=str, default="",
                        help="Path for the optimization results CSV. Defaults to a timestamped file in results/.")
    parser.add_argument("--sample_size", type=int, default=3000)
    parser.add_argument("--gpu_num", type=int, default=8)

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dcr_sft_batch_size", type=int, default=16)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dcr_sft_data", type=str, default="")
    parser.add_argument("--spa_sft_data", type=str, default="")
    parser.add_argument("--max_steps", type=int, default=2000000)
    parser.add_argument("--save_steps", type=int, default=20000)

    args = parser.parse_args()
    return args
