import datetime

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

from utils import get_parse, set_logger, record_operating_parameters, setup_seed


def process_func(example):
    MAX_LENGTH = 512
    output_group_length = 128
    input_group_length = MAX_LENGTH - output_group_length

    p = ''

    obj = example['obj']
    if obj == 'qed&drd2':
        property_t = eval(example['property'])
        properties = (round(property_t[0], 2), round(property_t[1], 2))
    else:
        properties = round(float(example['property'].replace('"', '')), 2)

    instruction_input = example['smiles'] + '<|>' + obj + '=' + str(properties) + '<|>'
    response_output = example['mask_smiles']

    instruction = tokenizer(
        "<|im_start|>system\n" + p + "<|im_end|>\n<|im_start|>user\n" + instruction_input + "<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
        truncation=True,
        max_length=input_group_length
    )

    response = tokenizer(
        response_output + "<|im_end|>",
        add_special_tokens=False,
        truncation=True,
        max_length=output_group_length
    )

    input_ids = instruction['input_ids'] + response['input_ids'] + [tokenizer.pad_token_id]
    attention_mask = instruction['attention_mask'] + response['attention_mask'] + [1]
    labels = [-100] * len(instruction['input_ids']) + response['input_ids'] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels}


logger = set_logger('spa_sft')

args = get_parse()
record_operating_parameters(logger, args)
checkpoint_save_path = datetime.datetime.now().strftime("%Y%m%d%H")

setup_seed(args.seed)

print('Loading tokenizer')
tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False,
                                          trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print('Loading model')
model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto",
                                             torch_dtype=torch.bfloat16)
model.enable_input_require_grads()

print('Loading dataset')
df = pd.read_csv(args.spa_sft_data)
ds = Dataset.from_pandas(df)
ds = ds.filter(lambda x: x['property'] is not None and x['obj'] is not None)
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

training_args = TrainingArguments(
    output_dir="spa_sft_model/" + checkpoint_save_path,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    logging_steps=100,
    max_steps=200000,
    save_steps=5000,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    remove_unused_columns=False,
    report_to=["tensorboard"],
    per_device_train_batch_size=args.batch_size
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()
trainer.save_model(training_args.output_dir)
