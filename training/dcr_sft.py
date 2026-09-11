import datetime

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

from utils import get_parse, set_logger, record_operating_parameters, setup_seed, get_prompt


def process_func(example):
    MAX_LENGTH = 512
    output_group_length = 128
    input_group_length = MAX_LENGTH - output_group_length

    p = get_prompt('dcr_prompt')
    if args.use_one_mask:
        p = get_prompt('dcr_one_mask_prompt_f2')

    instruction_input = example['instruction'] + example['input']
    response_output = example['output']
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

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


logger = set_logger('dcr_sft')

args = get_parse()
record_operating_parameters(logger, args)
checkpoint_save_path = datetime.datetime.now().strftime("%Y%m%d%H")

setup_seed(args.seed)

print('Loading tokenizer')
tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False,
                                          trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print('Loading model')
model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
model.enable_input_require_grads()

print('Loading dataset')
df = pd.read_json(args.dcr_sft_data)
ds = Dataset.from_pandas(df)
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

training_args = TrainingArguments(
    output_dir="dcr_sft_model/" + checkpoint_save_path,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    logging_steps=100,
    max_steps=args.max_steps,
    save_steps=args.save_steps,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    remove_unused_columns=False,
    report_to=["tensorboard"],
    per_device_train_batch_size=args.dcr_sft_batch_size
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()
trainer.save_model(training_args.output_dir)
