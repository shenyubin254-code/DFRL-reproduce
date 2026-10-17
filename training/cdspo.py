import datetime
import os
import random

import torch
from datasets import load_dataset, concatenate_datasets
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from trl import GRPOConfig, GRPOTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback
)
from rdkit import RDLogger

from utils import get_parse, set_logger, record_operating_parameters, setup_seed, get_prompt, \
    drd2_predict_oracle
from reward_function import dcr_grpo_reward, dcr_format_reward

os.environ["TOKENIZERS_PARALLELISM"] = "false"
RDLogger.DisableLog('rdApp.*')

SPA_GRPO_DATA_DIR = os.environ.get("SPA_GRPO_DATA_DIR", "data/2mask_grpo")
DCR_GRPO_DATA_DIR = os.environ.get("DCR_GRPO_DATA_DIR", "data/env_grpo")

STEPS_PER_STAGE = 10000
STEPS_PER_SWITCH = 500
NUM_STAGES = 5

CURR_LISTS = [
    [80000, 50000, 30000, 30000, 10000],
    [60000, 50000, 40000, 30000, 20000],
    [40000, 40000, 40000, 40000, 40000],
    [20000, 30000, 40000, 50000, 60000],
    [10000, 20000, 30000, 50000, 90000],
]


class AlternatingStopCallback(TrainerCallback):
    def __init__(self, interval):
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        if (
            state.global_step > 0
            and state.global_step % self.interval == 0
            and state.global_step < state.max_steps
        ):
            control.should_save = True
            control.should_training_stop = True

        return control


def spa_prompt(x):
    p = ''

    smiles = x['smiles']
    obj = x['obj']

    if obj == 'drd2modof':
        obj_list = ['drd2', 'qed&drd2']
        obj = random.choice(obj_list)

    if obj == 'qed&drd2':
        pro = (x['qed'], x['drd2'])
    else:
        pro = x[x['obj']]

    input_smiles = smiles + '<|>' + obj + '=' + str(pro) + '<|>'
    targets = smiles + '<|>' + obj + '<|>'

    return {
        "prompt": [{'role': 'system', 'content': p},
                   {'role': 'user', 'content': input_smiles}],
        'targets': targets}


def spa_environment_reward_combine(completions, targets, **kwargs):
    rewards = []

    completion_contents = [c[0]["content"] if isinstance(c, list) else c for c in completions]

    dcr_input_texts = []
    valid_indices = []
    parse_cache = []

    for i, (content, target_str) in enumerate(zip(completion_contents, targets)):
        try:
            if '<|mask|>' not in content:
                parse_cache.append(None)
                continue

            parts = target_str.split('<|>')
            if len(parts) < 2:
                parse_cache.append(None)
                continue

            original_smiles, obj = parts[0], parts[1]

            mol_origin = Chem.MolFromSmiles(original_smiles)
            if not mol_origin:
                parse_cache.append(None)
                continue

            base_props = {
                'qed': QED.qed(mol_origin),
                'plogp': Descriptors.MolLogP(mol_origin),
                'drd2': drd2_predict_oracle([original_smiles])
            }

            parse_cache.append({
                'obj': obj,
                'base': base_props,
                'original_smiles': original_smiles
            })

            dcr_prompt_str = content + '<|>' + obj + '<|>'

            messages = [
                {"role": "system", "content": get_prompt('dcr_one_mask_prompt_f2')},
                {"role": "user", "content": dcr_prompt_str}
            ]
            text = dcr_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            dcr_input_texts.append(text)
            valid_indices.append(i)

        except Exception as e:
            print(f"Preprocessing error at index {i}: {e}")
            parse_cache.append(None)

    final_rewards = [0.0] * len(completions)
    generated_fragments = []

    if len(dcr_input_texts) > 0:
        try:
            if dcr_tokenizer.pad_token_id is None:
                dcr_tokenizer.pad_token_id = dcr_tokenizer.eos_token_id

            inputs = dcr_tokenizer(
                dcr_input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to('cuda')

            with torch.no_grad():
                dcr_model.eval()
                generated_ids = dcr_model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=dcr_tokenizer.pad_token_id,
                    eos_token_id=dcr_tokenizer.eos_token_id
                )

            gen_sequences = generated_ids[:, inputs.input_ids.shape[-1]:]
            generated_fragments = dcr_tokenizer.batch_decode(gen_sequences, skip_special_tokens=True)

        except Exception as e:
            print(f"DCR model batch generation failed: {e}")
            return final_rewards

    gen_ptr = 0
    for i in range(len(completions)):
        if i not in valid_indices:
            continue

        try:
            fragment = generated_fragments[gen_ptr]
            gen_ptr += 1

            content = completion_contents[i]
            meta = parse_cache[i]
            if meta is None:
                continue

            full_smiles = content.replace('<|mask|>', fragment)

            mol_new = Chem.MolFromSmiles(full_smiles)
            if not mol_new:
                final_rewards[i] = 0.0
                continue

            qed_new = QED.qed(mol_new)
            logp_new = Descriptors.MolLogP(mol_new)
            drd2_new = drd2_predict_oracle([full_smiles])

            obj = meta['obj']
            base = meta['base']

            is_improved = False
            r = 0.0

            if obj == 'qed':
                if qed_new > base['qed']:
                    r = 1.0 + qed_new - base['qed']
            elif obj == 'plogp':
                if logp_new > base['plogp']:
                    r = 1.0 + logp_new - base['plogp']
            elif obj == 'drd2':
                if drd2_new > base['drd2']:
                    r = 1.0 + drd2_new - base['drd2']
            elif obj == 'qed&drd2':
                qed_th = 0.6
                drd2_th = 0.5

                eff_qed_new = min(qed_new, qed_th)
                eff_drd2_new = min(drd2_new, drd2_th)
                score_new = eff_qed_new + eff_drd2_new

                eff_qed_base = min(base['qed'], qed_th)
                eff_drd2_base = min(base['drd2'], drd2_th)
                score_base = eff_qed_base + eff_drd2_base

                if score_new > score_base:
                    r = 1.0 + (score_new - score_base)
                    is_improved = True

                if qed_new > qed_th and drd2_new > drd2_th:
                    bonus = 0.5 + 0.5 * ((qed_new - qed_th) + (drd2_new - drd2_th))

                    if is_improved:
                        r += bonus
                    else:
                        raw_sum_new = qed_new + drd2_new
                        raw_sum_base = base['qed'] + base['drd2']
                        if raw_sum_new > raw_sum_base:
                            r = 1.0 + bonus
                        else:
                            r = 1.0

            final_rewards[i] = r

        except Exception as e:
            print(f"Reward calculation error at index {i}: {e}")
            final_rewards[i] = 0.0

    return final_rewards


def spa_format_reward(completions, targets, **kwargs):
    rewards = []
    completion_contents = [completion[0]["content"] for completion in completions]
    for completion_content, target in zip(completion_contents, targets):
        if '<|mask|>' in completion_content:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def check_token_list(input_string, word_table):
    for char in input_string:
        if char not in word_table:
            return False
    else:
        return True


def spa_token_reward(completions, targets, **kwargs):
    rewards = []
    completion_contents = [completion[0]["content"] for completion in completions]
    tokens = ['#', '(', ')', '+', '-', '1', '2', '3', '4', '5', '6', '7', '=', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P',
              'S', '[', ']', 'l', 'r']
    for completion_content, target in zip(completion_contents, targets):
        try:
            completion_content = completion_content.replace('<|mask|>', '')
            if check_token_list(completion_content, tokens):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception as e:
            rewards.append(0.0)
    return rewards


def spa_len_reward(completions, targets, **kwargs):
    rewards = []
    completion_contents = [completion[0]["content"] for completion in completions]
    for completion_content, target in zip(completion_contents, targets):
        try:
            if len(completion_content) > 100:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception as e:
            rewards.append(0.0)
    return rewards


logger = set_logger('cdspo')

args = get_parse()
record_operating_parameters(logger, args)
checkpoint_save_path = datetime.datetime.now().strftime("%Y%m%d%H")

setup_seed(args.seed)

print('Loading tokenizer')
spa_tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False,
                                              trust_remote_code=True)
dcr_tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False,
                                              trust_remote_code=True, padding_side='left')

print('Loading models')
dcr_model = AutoModelForCausalLM.from_pretrained(args.dcr_model_path, torch_dtype=torch.bfloat16).to('cuda')
dcr_model.enable_input_require_grads()
dcr_model.requires_grad_(True)

spa_model = AutoModelForCausalLM.from_pretrained(args.spa_model_path, torch_dtype=torch.bfloat16).to('cuda')
spa_model.enable_input_require_grads()
spa_model.requires_grad_(True)

curriculum_datasets = [
    load_dataset('csv', data_files=f'{SPA_GRPO_DATA_DIR}/1sampled_data_0.csv', split='train'),
    load_dataset('csv', data_files=f'{SPA_GRPO_DATA_DIR}/1sampled_data_1.csv', split='train'),
    load_dataset('csv', data_files=f'{SPA_GRPO_DATA_DIR}/1sampled_data_2.csv', split='train'),
    load_dataset('csv', data_files=f'{SPA_GRPO_DATA_DIR}/1sampled_data_3.csv', split='train'),
    load_dataset('csv', data_files=f'{SPA_GRPO_DATA_DIR}/1sampled_data_4.csv', split='train')
]

dataset_dcr = load_dataset('json', data_files=f'{DCR_GRPO_DATA_DIR}/env_grpo.json', split='train')
dataset_dcr = dataset_dcr.map(lambda x: {
    "prompt": [{'role': 'system', 'content': get_prompt('dcr_prompt')},
               {'role': 'user', 'content': x['instruction']}],
    'targets': x['output']})
dcr_train_test = dataset_dcr.train_test_split(test_size=0.1)

num_blocks_per_stage = STEPS_PER_STAGE // STEPS_PER_SWITCH

for stage_idx, curr_list in enumerate(CURR_LISTS):
    print(f"=== Starting Curriculum Stage {stage_idx + 1}/{NUM_STAGES} ===")

    current_stage_datasets = []
    for nums in range(5):
        req_count = curr_list[nums]
        total_count = len(curriculum_datasets[nums])
        actual_count = min(req_count, total_count)
        subset = curriculum_datasets[nums].shuffle(seed=42 + stage_idx).select(range(actual_count))
        current_stage_datasets.append(subset)
    dataset_spa = concatenate_datasets(current_stage_datasets)
    print(f"Stage {stage_idx + 1} Data Distribution: {curr_list} | Total: {len(dataset_spa)}")

    dataset_spa = dataset_spa.map(lambda x: spa_prompt(x))
    spa_train_test = dataset_spa.train_test_split(test_size=0.1)

    spa_output_dir = os.path.join("spa_grpo_model", f"{checkpoint_save_path}_stage_{stage_idx}")

    spa_training_args = GRPOConfig(
        output_dir=spa_output_dir,
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
        logging_steps=100,
        max_steps=STEPS_PER_STAGE,
        save_strategy="steps",
        save_steps=STEPS_PER_SWITCH,
        save_only_model=False,
        save_total_limit=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_prompt_length=512,
        max_completion_length=128,
        beta=0.001,
        temperature=1.5,
        per_device_train_batch_size=20,
        num_generations=20,
        report_to=["tensorboard"]
    )
    trainer_spa = GRPOTrainer(
        processing_class=spa_tokenizer,
        model=spa_model,
        reward_funcs=[spa_environment_reward_combine, spa_token_reward, spa_format_reward, spa_len_reward],
        args=spa_training_args,
        train_dataset=spa_train_test['train'],
        eval_dataset=spa_train_test['test'],
        peft_config=None,
        callbacks=[
            AlternatingStopCallback(STEPS_PER_SWITCH)
        ],
    )

    dcr_training_args = GRPOConfig(
        output_dir=os.path.join(
            "dcr_grpo_model",
            f"{checkpoint_save_path}_stage_{stage_idx}"
        ),
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        logging_steps=10,
        max_steps=STEPS_PER_STAGE,
        save_strategy="steps",
        save_steps=STEPS_PER_SWITCH,
        save_only_model=False,
        save_total_limit=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_prompt_length=1280,
        max_completion_length=1280,
        per_device_train_batch_size=8,
        num_generations=8,
        beta=0.001,
        report_to=["tensorboard"]
    )
    trainer_dcr = GRPOTrainer(
        processing_class=dcr_tokenizer,
        model=dcr_model,
        reward_funcs=[dcr_format_reward, dcr_grpo_reward],
        args=dcr_training_args,
        train_dataset=dcr_train_test['train'],
        eval_dataset=dcr_train_test['test'],
        peft_config=None,
        callbacks=[
            AlternatingStopCallback(STEPS_PER_SWITCH)
        ],
    )

    spa_checkpoint = None
    dcr_checkpoint = None

    for block in range(num_blocks_per_stage):
        target_step = (block + 1) * STEPS_PER_SWITCH

        # ===== Train SPA =====
        dcr_model.requires_grad_(False)
        dcr_model.eval()

        spa_model.requires_grad_(True)
        spa_model.train()

        trainer_spa.train(
            resume_from_checkpoint=spa_checkpoint
        )

        if trainer_spa.state.global_step != target_step:
            raise RuntimeError(
                f"SPA step mismatch: expected {target_step}, "
                f"got {trainer_spa.state.global_step}"
            )

        spa_checkpoint = os.path.join(
            spa_training_args.output_dir,
            f"checkpoint-{target_step}"
        )

        if not os.path.isdir(spa_checkpoint):
            raise FileNotFoundError(spa_checkpoint)

        # ===== Train DCR =====
        spa_model.requires_grad_(False)
        spa_model.eval()

        dcr_model.requires_grad_(True)
        dcr_model.train()

        trainer_dcr.train(
            resume_from_checkpoint=dcr_checkpoint
        )

        if trainer_dcr.state.global_step != target_step:
            raise RuntimeError(
                f"DCR step mismatch: expected {target_step}, "
                f"got {trainer_dcr.state.global_step}"
            )

        dcr_checkpoint = os.path.join(
            dcr_training_args.output_dir,
            f"checkpoint-{target_step}"
        )

        if not os.path.isdir(dcr_checkpoint):
            raise FileNotFoundError(dcr_checkpoint)

    trainer_spa.save_model(spa_training_args.output_dir)
    trainer_dcr.save_model(dcr_training_args.output_dir)
