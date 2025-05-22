import base64
import os
import joblib
import numpy as np
from openai import OpenAI, AsyncOpenAI
from clip_similarity import calculate_clip_similarity
from model import DeepRegressionModel
from prompts import system_prompt_few_shot_learning, multi_expert, system_prompt_evaluator, multi_expert_enhanced_prompt, system_prompt_evaluator_enhanced
import torch

from pydantic import BaseModel,Field
class Model(BaseModel):
    reasoning: str = Field(...,
        description="Explain the step-by-step thought process behind the provided values. Include key considerations and how they influenced the final decisions."
    )

    alignment_score: float




client = AsyncOpenAI()
def llm_score_few_shot_learning(ground_truth_caption_with_score:str, ground_truth_image : str, caption:str, image:str):
    #print(f"ground_truth_example: {ground_truth_example}")
    #ground_truth_prompt_with_score = ground_truth_example["documents"][0]
    print(f"ground_truth_prompt_with_score: {(ground_truth_caption_with_score)}, Prompt: {caption}")
    #print(f"IMAGE: {ground_truth_image}")
    actual_bytes = eval(ground_truth_image[0]["image"])
    
    # Convert bytes to base64
    image_base64_ground_truth = base64.b64encode(actual_bytes).decode('utf-8')
    
    
    image_base64 = base64.b64encode(image).decode("utf-8")
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        temperature=0.0,
        response_format=Model,
        messages=[
            {
                "role": "system",
                "content": [{
                    "type":"text", 
                    "text": system_prompt_few_shot_learning
            }],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{ground_truth_caption_with_score}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64_ground_truth}"},
                    },
                    
                    {
                        "type": "text",
                        "text": f"Textual prompt: {caption}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ],
    )
    output = response.choices[0].message.parsed
    #print(f"Output: {output}")
    
    return {
            "alignment_score": float(output.alignment_score), 
            "artifact_score": float(output.artifact_score), 
            "aesthetics_score": float(output.aesthetics_score)
            }


from pydantic import BaseModel,Field
from pydantic import BaseModel,Field
class ModelMultiExpert(BaseModel):
    reasoning_of_expert_1: str = Field(...,
        description="Explanation of EXPERT 1 the step-by-step thought process behind the provided values. Include key considerations and how they influenced the final decisions."
    )
    reasoning_of_expert_2: str = Field(...,
        description="Explanation of EXPERT 2 the step-by-step thought process behind the provided values. Include key considerations and how they influenced the final decisions."
    )
    reasoning_of_expert_3: str = Field(...,
        description="Explanation of EXPERT 3 the step-by-step thought process behind the provided values. Include key considerations and how they influenced the final decisions."
    )

    reasoning_of_final_expert: str = Field(...,
        description="Explanation of EXPERT 4 the step-by-step thought process behind the provided values. Include key considerations and how they influenced the final decisions."
    )
    final_alignment_score: float


async def llm_score_multi_expert(image, caption):
    image_base64 = base64.b64encode(image).decode("utf-8")
    #print(f"Base 64: {image_base64}")
    response = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        temperature=0.0,
        top_p=1.0,
        response_format=ModelMultiExpert,
        messages=[
            {
                "role": "system",
                "content": [{
                    "type":"text", 
                    "text": multi_expert
            }],
            },
            {
                "role": "user",
                "content": [
                        {
                        "type": "text",
                        "text": f"Text prompt: {caption}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },

                ],
            }
        ],
    )
    output = response.choices[0].message.parsed
    #print(f"Output: {output}")
    
    return {"alignment_score": output.final_alignment_score}



async def naive_evaluation(image, caption):
    image_base64 = base64.b64encode(image).decode("utf-8")
    #print(f"Base 64: {image_base64}")
    response = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        temperature=0.0,
        response_format=Model,
        messages=[
            {
                "role": "system",
                "content": [{
                    "type":"text", 
                    "text": system_prompt_evaluator_enhanced
            }],
            },
            {
                "role": "user",
                "content": [
                        {
                        "type": "text",
                        "text": f"Text prompt: {caption}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },

                ],
            }
        ],
    )
    output = response.choices[0].message.parsed
    return {"alignment_score": output.alignment_score}


async def classic_evaluation(image, caption):
    image_base64 = base64.b64encode(image).decode("utf-8")
    #print(f"Base 64: {image_base64}")
    response = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        temperature=0.0,
        response_format=Model,
        messages=[
            {
                "role": "system",
                "content": [{
                    "type":"text", 
                    "text":system_prompt_evaluator_enhanced
            }],
            },
            {
                "role": "user",
                "content": [
                        {
                        "type": "text",
                        "text": f"Text prompt: {caption}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },

                ],
            }
        ],
    )
    output = response.choices[0].message.parsed
    model = DeepRegressionModel(input_dim=2)
    model.load_state_dict(torch.load("/model_2/model/deep_regression_model_v2.pth"))
    model.eval()
    scaler = joblib.load('/model_2/model/scaler_v2.pkl')
    clip_similarity = calculate_clip_similarity(image, caption)
    #print(f"LLM score: {output.alignment_score}, {clip_similarity}")
    sample = torch.FloatTensor(np.array([output.alignment_score, clip_similarity]))
    scaled_sample = scaler.transform(sample.reshape(1, -1))
    scaled_features_tensor = torch.tensor(scaled_sample, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(scaled_features_tensor)
        
    prediction = prediction.numpy().item()
    return {"alignment_score": prediction}


import base64
from openai import OpenAI
import os



class ModelEnhanced(BaseModel):
    reasoning_of_expert_1: str = Field(...,
        description="Explanation of EXPERT 1 the step-by-step thought process behind the provided values. Include key considerations and how they influenced the\
              final decisions."
    )
    reasoning_of_expert_2: str = Field(...,
        description="Explanation of EXPERT 2 the step-by-step thought process behind the provided values. Include key considerations and how they influenced the\
              final decisions."
    )
    reasoning_of_expert_3: str = Field(...,
        description="Explanation of EXPERT 3 the step-by-step thought process behind the provided values. Include key considerations and how they influenced the\
              final decisions."
    )

    reasoning_of_expert_4: str = Field(...,
        description="Explanation of EXPERT 4 the step-by-step thought process behind the provided values. Include key considerations and how they influenced \
            the final decisions."
    )
    reasoning_of_expert_5: str = Field(...,
        description="Explanation of EXPERT 5 the step-by-step thought process behind the provided values. Include key considerations and how\
              they influenced the final decisions."
    )
    score_of_expert_1: float | None
    score_of_expert_2: float | None
    score_of_expert_3: float | None
    score_of_expert_4: float | None
    score_of_expert_5: float | None

async def multi_expert_enhanced(image, caption):
    image_base64 = base64.b64encode(image).decode("utf-8")
    #print(f"Base 64: {image_base64}")
    response = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        temperature=0.0,
        top_p=1.0,
        response_format=ModelEnhanced,
        messages=[
            {
                "role": "system",
                "content": [{
                    "type":"text", 
                    "text": multi_expert_enhanced_prompt
            }],
            },
            {
                "role": "user",
                "content": [
                        {
                        "type": "text",
                        "text": f"Text prompt: {caption}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail":"high"},
                    },

                ],
            }
        ],
    )
    output = response.choices[0].message.parsed
    score_1,score_2,score_3,score_4, score_5 = output.score_of_expert_1, output.score_of_expert_2,  output.score_of_expert_3, output.score_of_expert_4, output.score_of_expert_5
    
    return {"alignment_score":np.mean([score_1,score_2, score_3, score_4, score_5]), "Score 1": score_1, "Score 2": score_2, "Score 3": score_3, "Score 4": score_4, "Score 5": score_5}




class Model(BaseModel):
    reasoning: str = Field(...,
        description="Explain the step-by-step thought process behind the provided values. Include key considerations and how they influenced the final decisions."
    )
    alignment_score: float




def llm_score_classic(caption, image):
    prompt = caption
    image_base64 = base64.b64encode(image).decode("utf-8")
    #print(f"Base 64: {image_base64}")
    client = OpenAI()

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        temperature=0.0,
        response_format=Model,
        messages=[
            {
                "role": "system",
                "content": [{
                    "type":"text", 
                    "text": system_prompt_evaluator
            }],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Textual prompt: {prompt}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ],
    )
    output = response.choices[0].message.parsed
    model = DeepRegressionModel(input_dim=2)
    model.load_state_dict(torch.load("/model/model_2/deep_regression_model_v2.pth"))
    model.eval()
    scaler = joblib.load('/model/model_2/scaler_v2.pkl')
    clip_similarity = calculate_clip_similarity(image, prompt)
    sample = torch.FloatTensor(np.array([output.alignment_score, clip_similarity]))
    scaled_sample = scaler.transform(sample.reshape(1, -1))
    scaled_features_tensor = torch.tensor(scaled_sample, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(scaled_features_tensor)
        
    prediction = prediction.numpy().item()
    return {"alignment_score": prediction}