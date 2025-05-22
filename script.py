import modal
import wandb

from utils import parse_record, search_for_image

app = modal.App("example-get-started")

from tqdm import tqdm
import tensorflow as tf
from datasets import Dataset
from llm_score import llm_score_few_shot_learning
from datasets import load_dataset
from evaluator import parse_tfrecord_file, parse_tfrecord_file_multi_expert, parse_tfrecord_file_clip_simuilarity, parse_tfrecord_file_qwen_evaluation

vllm_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install('git')
    .pip_install(
        "datasets",
        "transformers",
        "torch",
        'torchvision',
        "pillow",
        "tensorflow",
        "scipy",
        "openai",
        "pysqlite3-binary",
        "sentence_transformers",
        'tf-keras',
        "chromadb",
        "joblib",
        "qwen_vl_utils",
        'qwen-vl-utils[decord]==0.0.8',
        'accelerate',
        'wandb',
        'peft',
        'trl',
        'bitsandbytes',
        "einops"
    )
)


vol = modal.Volume.from_name("dataset")
model = modal.Volume.from_name('model_2')
        
import re


import base64
from openai import OpenAI
import os

from pydantic import BaseModel, Field
from prompt import system_prompt_evaluator



import base64
from openai import OpenAI
import os

@app.function(volumes={"/dataset": vol, "/model_2": model}, gpu="A100-40GB", image=vllm_image,  timeout=60 * 60 * 3)
def evaluate_few_shot_learning():
    from PIL import Image
    import asyncio
    from datasets import load_from_disk

    filename = "/dataset/dataset/train.tfrecord"
    ds = load_from_disk(dataset_path="/dataset/train")
    print("here")
    json_list = [dict(example) for example in ds]
    print("list processed")

    llm_score_results, human_scores = asyncio.run(parse_tfrecord_file_qwen_evaluation(json_list=json_list, filename=filename))
    alignment_llm_score = []
    alignment_human_score = []

    print(f"LLM score: {llm_score_results}\nHuman score: {human_scores}")

    for llm_score, human_score in zip(llm_score_results, human_scores):
        alignment_llm_score.append(llm_score["alignment_score"])
        

        alignment_human_score.append(human_score["alignment_score"])

    #with open('/dataset/train_dataset/detailed_evaluation.txt', 'w') as f:
    #    f.write(f"Alignment score LLM: {llm_score_results}\n")
    #
    #    f.write("\n==========================\n")


    import scipy.stats as stats
    pearsonr_alignment, _ = stats.pearsonr(alignment_llm_score, alignment_human_score)
    spearmanr_alignment,_ = stats.spearmanr(alignment_llm_score, alignment_human_score)



    print(f"Pearsonr alignment : {pearsonr_alignment}, Spearmanr alignment: {spearmanr_alignment}")


@app.function(volumes={"/dataset": vol, "/model": model}, image=vllm_image,  timeout=60 * 60 * 3)
def calculate_clip_similarity():
    import torch
    from PIL import Image
    import requests
    from transformers import CLIPProcessor, CLIPModel, CLIPConfig
    import io
    import tensorflow as tf

    from datasets import load_dataset, load_from_disk

    filename = "/dataset/dataset/test.tfrecord"
    ds = load_from_disk(dataset_path="/test/train")
    json_list = [dict(example) for example in ds]
    print("list processed")

    parse_tfrecord_file_clip_simuilarity(json_list=json_list, filename=filename)
    

    

@app.function(volumes={"/dataset": vol}, image=vllm_image,  timeout=60 * 60*2,)
def download_dataset():
    from PIL import Image
    import tensorflow as tf
    from datasets import Dataset
    from datasets import load_dataset
    import urllib3, socket
    from urllib3.connection import HTTPConnection

    HTTPConnection.default_socket_options = ( 
        HTTPConnection.default_socket_options + [
        (socket.SOL_SOCKET, socket.SO_SNDBUF, 2000000), 
        (socket.SOL_SOCKET, socket.SO_RCVBUF, 2000000)
    ])
    import aiohttp
    dataset = load_dataset("yuvalkirstain/pickapic_v1", split="train", cache_dir="/dataset/cache", streaming=True,  storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600*2)}})
    data_list = list(dataset.take(200000))  
    local_dataset = Dataset.from_list(data_list)

    # Save it locally
    local_dataset.save_to_disk("local_dataset")
    output_dir = "/dataset/train"
    
    local_dataset.save_to_disk(output_dir)
    vol.commit()
 

    return None



@app.function(volumes={"/dataset": vol, "/model_2": model}, gpu="A100-40GB", image=vllm_image,  timeout=60 * 60 * 3)
def evaluate_qwen():
    from PIL import Image
    import asyncio
    from datasets import load_from_disk

    filename = "/dataset/dataset/test.tfrecord"
    ds = load_from_disk(dataset_path="/dataset/test")
    print("here")
    json_list = [dict(example) for example in ds]
    print("list processed")

    llm_score_results, human_scores = asyncio.run(parse_tfrecord_file_qwen_evaluation(json_list=json_list, filename=filename))
    alignment_llm_score = []
    alignment_human_score = []

    print(f"LLM score: {llm_score_results}\nHuman score: {human_scores}")

    for llm_score, human_score in zip(llm_score_results, human_scores):
        alignment_llm_score.append(llm_score["alignment_score"])
        

        alignment_human_score.append(human_score["alignment_score"])

    #with open('/dataset/train_dataset/detailed_evaluation.txt', 'w') as f:
    #    f.write(f"Alignment score LLM: {llm_score_results}\n")
    #
    #    f.write("\n==========================\n")


    import scipy.stats as stats
    pearsonr_alignment, _ = stats.pearsonr(alignment_llm_score, alignment_human_score)
    spearmanr_alignment,_ = stats.spearmanr(alignment_llm_score, alignment_human_score)



    print(f"Pearsonr alignment : {pearsonr_alignment}, Spearmanr alignment: {spearmanr_alignment}")

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


@app.function(volumes={"/dataset": vol, "/model_2": model}, image=vllm_image,  timeout=60 * 60 * 3)
def train_qwen():
    process_list()

@app.local_entrypoint()
def main():
    evaluate_few_shot_learning.remote()