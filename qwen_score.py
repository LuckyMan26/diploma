import base64
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from prompts import system_prompt_evaluator
import json
class Qwen(object):
    def __init__(self, device: str = "cuda", max_length: int = 2048):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)       
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    def score(self, prompt: str, image: str) -> float:
        image_base64 = base64.b64encode(image).decode("utf-8")
        messages = [
        
            {"role": "system",
            "content": system_prompt_evaluator
            },
            {"role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"data:image;base64,{image_base64}",
                },
                {"type": "text", "text": f"Prompt: {prompt}"},
            ],
            },
            {"content": "", 
             "role": "assistant"
             }
    ]
        text = self.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
        #print(f'Text: {text}')
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=512, temperature=0.001)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        output_text = output_text[0].replace("```json", '').replace("```", '')
        output_text = json.loads(output_text)
        return {"alignment_score": output_text["alignment_score"]}