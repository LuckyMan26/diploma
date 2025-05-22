import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, Dataset
from prompt import system_prompt_evaluator

import tensorflow as tf
from transformers import HfArgumentParser, TrainingArguments, set_seed
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration

from peft import LoraConfig

import torch
from evaluator import parse_record, search_for_image

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
def create_and_prepare_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto", 
    config=bnb_config,

)       
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    peft_config = LoraConfig(
        r=32,
        lora_alpha=128,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["k_proj", "q_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj", "lm_head", "linear"],
        modules_to_save=["embed_tokens", "lm_head", "input_layernorm", "post_attention_layernorm", "norm"]
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    #tokenizer.add_special_tokens({"pad_token": '<pad>'})

    #model.config.pad_token_id = tokenizer.pad_token_id
    #model.resize_token_embeddings(len(tokenizer))

    return model, peft_config, tokenizer


from datasets import load_from_disk

from datasets import Dataset, DatasetDict, Features, Value, Array3D, Image

def process_list(type="train"):
    filename = f"/dataset/dataset/{type}.tfrecord"
    ds = load_from_disk(dataset_path=f"/dataset/{type}")
    features = Features({
        "image": Value("string"),  # Automatically handles image loading and uploading
        "prompt": Value("string"),
        "human_alignment_score": Value("float32"),
    })

    json_list = [dict(example) for example in ds]
    batch_size = 8
    raw_dataset = tf.data.TFRecordDataset(filename)
    train_dataset = []
    for idx, raw_record in enumerate(raw_dataset):
        

       
        filename, uid, human_score = parse_record(raw_record)
        

        # Fetch image and prompt
        image, prompt = search_for_image(ds=json_list, uid=uid)
        if not image and not prompt:
            print("NO IMAGE FOUND")
        if image and prompt:
            train_dataset.append({"image": image, "prompt": prompt, "human_alignment_score": human_score["alignment_score"]})


        if len(train_dataset) > 5:
            hf_dataset = Dataset.from_list(train_dataset, features=features)
            dataset_repo = "LuckyMan123/diploma_dataset"
            hf_dataset.push_to_hub(dataset_repo, private=True, token="<token>")
            return train_dataset
        print(f"Idx: {idx}. Length {len(train_dataset)} of list")

def prepare_datasets_and_collator(tokenizer):
    dataset_train = process_list()
    dataset_valid = process_list(type="test")
    conversations_train = []
    for i in dataset_train:


        template = [
            {
            "role": "system",
            "content": system_prompt_evaluator
            },
            {
            "role": "user",
            "content": {"type": "image", "image": f"data:image;base64,{i['image']}", "text": f"Prompt: {i['prompt']}"},
            },
            {
            "role": "assistant",
            "content": i['human_alignment_score']
            }
        ]
        chat = tokenizer.apply_chat_template(template, tokenize=False).replace('<|im_start|>', '')

        conversations_train.append({"text": chat})
        
    print(f'EXAMPLE: {conversations_train[0]["text"]}')
    
    conversations_test = []
    for i in dataset_valid:

        template = [
            {
            "role": "system",
            "content": system_prompt_evaluator
            },
            {
            "role": "user",
            "content": {"type": "image", "image": f"data:image;base64,{i['image']}", "text": f"Prompt: {i['prompt']}"},
            },
            {
            "role": "assistant",
            "content": i['human_alignment_score']
            }
        ]
        chat = tokenizer.apply_chat_template(template, tokenize=False).replace('<|im_start|>', '')

        conversations_test.append({"text": chat})

    print(f'EXAMPLE: {conversations_test[0]["text"]}')

    
    conversations_train = conversations_train[:500]
    conversations_test = conversations_test[:200]

    from random import shuffle

    shuffle(conversations_train)
    shuffle(conversations_test)

    training_dataset = Dataset.from_list(conversations_train)
    validation_dataset = Dataset.from_list(conversations_test)

    response_template = 'assistant<|im_end|>'
    instruction_template = '<|im_start|>user'
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

    
    return training_dataset, validation_dataset, collator


def main(training_args):
    set_seed(training_args.seed)

    model, peft_config, tokenizer = create_and_prepare_model()
    train_dataset, eval_dataset, data_collator = prepare_datasets_and_collator(tokenizer)
    
    model.config.use_cache = False

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=False,
        dataset_text_field="text",
        max_seq_length=4096,
        data_collator=data_collator,
    )
    trainer.accelerator.print(f"{trainer.model}")
    trainer.model.print_trainable_parameters()

  
    trainer.train()
    trainer.save_model()



def train():

    a = TrainingArguments(
        bf16=True,
        ddp_find_unused_parameters=False,
        do_eval=True,
        eval_strategy="epoch",
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        learning_rate=2e-4,
        logging_strategy="steps",
        logging_steps=10,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        num_train_epochs=1.0,
        output_dir="output",
        per_device_eval_batch_size=4,
        per_device_train_batch_size=4,
        remove_unused_columns=True,
        report_to=['tensorboard', 'wandb'],
        save_strategy="steps",
        seed=100,
        warmup_ratio=0.05,
        weight_decay=0.0001,
        
    
    )

    main(a)
