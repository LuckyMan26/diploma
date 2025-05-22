import asyncio
from utils import parse_record,search_for_image
from llm_score import llm_score_few_shot_learning, llm_score_multi_expert, classic_evaluation, multi_expert_enhanced, llm_score_classic, naive_evaluation
import tensorflow as tf
from retriever import Retriever
from clip_similarity import calculate_clip_similarity, calculate_visual_embeddings
from qwen_score import Qwen

def parse_tfrecord_file(json_list, filename, batch_size=4):
    """Parses a TFRecord file, retrieves image and prompt data, and computes LLM scores."""
    llm_score_results = []
    human_scores = []
    retriever = Retriever()
    
    raw_dataset = tf.data.TFRecordDataset(filename).batch(batch_size)
    for idx, batch_records in enumerate(raw_dataset):
        batch_filenames, batch_images, batch_prompts, batch_human_scores = [], [], [], []

        for raw_record in batch_records:
            filename, uid, human_score = parse_record(raw_record)
            batch_filenames.append(filename)
            batch_human_scores.append(human_score)

            # Fetch image and prompt
            image, prompt = search_for_image(ds=json_list, uid=uid)
            batch_images.append(image)
            batch_prompts.append(prompt)

        # Compute LLM scores in batch
        examples = retriever.retrieve(batch_prompt=batch_prompts)
        #print(f"examples: {examples[0]['documents']}")
        print(f"Len: {len(examples['documents'])}, {len(examples['metadatas'])}")
        batch_llm_scores = [(llm_score_few_shot_learning(ground_truth_caption_with_score=ground_truth_caption,
                                                         ground_truth_image = ground_truth_image, 
                                                         caption=prompt, image=image)) 
                            for prompt, image,ground_truth_caption,ground_truth_image 
                            in zip(batch_prompts, batch_images,examples["documents"], examples["metadatas"])]
        #print(f"length: {len(batch_llm_scores)}")
        # Store results
        llm_score_results.extend(batch_llm_scores)
        human_scores.extend(batch_human_scores)

        print(f"Idx: {idx}. Processed {len(batch_filenames)} samples in batch.")
        


    return llm_score_results, human_scores

async def process_batch(batch_prompts, batch_images):
    
    tasks = [classic_evaluation(image, prompt) for prompt, image in zip(batch_prompts, batch_images)]
    results = await asyncio.gather(*tasks)
    return results

async def parse_tfrecord_file_multi_expert(json_list, filename, batch_size=8):
    """Parses a TFRecord file, retrieves image and prompt data, and computes LLM scores."""
    llm_score_results = []
    human_scores = []
    #retriever = Retriever()
    
    raw_dataset = tf.data.TFRecordDataset(filename).batch(batch_size)
    for idx, batch_records in enumerate(raw_dataset):
        batch_filenames, batch_images, batch_prompts, batch_human_scores = [], [], [], []

        for raw_record in batch_records:
            filename, uid, human_score = parse_record(raw_record)
            batch_filenames.append(filename)
            

            # Fetch image and prompt
            image, prompt = search_for_image(ds=json_list, uid=uid)
            if not image and not prompt:
                print("NO IMAGE FOUND")
            if image and prompt:
                batch_images.append(image)
                batch_prompts.append(prompt)
                batch_human_scores.append(human_score)

        try:
            batch_llm_scores = await (process_batch(batch_prompts, batch_images))

            print(f"length: {len(batch_llm_scores)}")

            llm_score_results.extend(batch_llm_scores)
            with open('/dataset/visual_embedding_score.txt', 'w') as f:
                f.write(str(llm_score_results))

            human_scores.extend(batch_human_scores)
            #with open('/dataset/progress_human_score_classis_evaluation.txt', 'w') as f:
                #f.write(str(human_scores))
        except Exception as e:
            print(f"Error processing record {idx}: {e}")
            continue

        print(f"Idx: {idx}. Length {len(llm_score_results)} of list")
        


    return llm_score_results, human_scores




def parse_tfrecord_file_clip_simuilarity(json_list, filename, batch_size=8):
    """Parses a TFRecord file, retrieves image and prompt data, and computes LLM scores."""
    batch_clip_score_results = []
    human_scores = []
    #retriever = Retriever()
    
    raw_dataset = tf.data.TFRecordDataset(filename).batch(batch_size)
    for idx, batch_records in enumerate(raw_dataset):
        batch_filenames, batch_images, batch_prompts, batch_human_scores = [], [], [], []

        for raw_record in batch_records:
            filename, uid, human_score = parse_record(raw_record)
            batch_filenames.append(filename)
            

            # Fetch image and prompt
            image, prompt = search_for_image(ds=json_list, uid=uid)
            if not image and not prompt:
                print("NO IMAGE FOUND")
            if image and prompt:
                batch_images.append(image)
                batch_prompts.append(prompt)
                batch_human_scores.append(human_score)


        batch_clip_scores = [calculate_clip_similarity(image, prompt) for image,prompt in zip(batch_images, batch_prompts)]

        print(f"length: {len(batch_clip_scores)}")

        batch_clip_score_results.extend(batch_clip_scores)
        with open('/dataset/progress_clip_score.txt', 'w') as f:
            f.write(str(batch_clip_score_results))

        human_scores.extend(batch_human_scores)
        with open('/dataset/progress_human_score_clip.txt', 'w') as f:
            f.write(str(human_scores))
        if len(batch_clip_score_results) > 1200:
            return batch_clip_score_results, human_scores
        print(f"Idx: {idx}. Length {len(batch_clip_score_results)} of list")
        


    return batch_clip_score_results, human_scores




async def parse_tfrecord_file_qwen_evaluation(json_list, filename):
    """Parses a TFRecord file, retrieves image and prompt data, and computes LLM scores."""
    llm_score_results = []
    human_scores = []
    #retriever = Retriever()
    qwen_evaluator = Qwen()
    raw_dataset = tf.data.TFRecordDataset(filename)
    for idx, batch_record in enumerate(raw_dataset):
        batch_filenames, batch_images, batch_prompts, batch_human_scores = [], [], [], []
        if idx > -1:
       
            filename, uid, human_score = parse_record(batch_record)
            batch_filenames.append(filename)
            

            # Fetch image and prompt
            image, prompt = search_for_image(ds=json_list, uid=uid)
            if not image and not prompt:
                print("NO IMAGE FOUND")
            if image and prompt:
                batch_images.append(image)
                batch_prompts.append(prompt)
                batch_human_scores.append(human_score)

            try:
                score = qwen_evaluator.score(prompt, image)
                #print(f"length: {len(score)}")
                llm_score_results.append(score)
                with open('/dataset/progress_qwen_7b_train.txt', 'w') as f:
                    f.write(str(llm_score_results))

                human_scores.extend(batch_human_scores)
            except Exception as e:
                print(f"Error processing record {idx}: {e}")
                continue
            #with open('/dataset/progress_human_score_classis_evaluation.txt', 'w') as f:
                #f.write(str(human_scores))
        if len(llm_score_results) > 1200:
            return llm_score_results, human_scores
        print(f"Idx: {idx}. Length {len(llm_score_results)} of list")
        


    return llm_score_results, human_scores