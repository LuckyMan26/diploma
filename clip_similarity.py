from transformers import CLIPProcessor, CLIPModel, CLIPConfig
from PIL import Image
import io
import torch
def calculate_clip_similarity(image_path, text, model_name="openai/clip-vit-base-patch32"):
    """
    Calculate the CLIP similarity score between an image and text.
    
    Args:
        image_path: Path to image file or URL
        text: Text to compare with the image
        model_name: CLIP model to use
        
    Returns:
        similarity_score: Cosine similarity between image and text embeddings
    """
    # Load model and processor
    config = CLIPConfig.from_pretrained(model_name)
    #config.text_config.max_position_embeddings = 512

    
    model = CLIPModel.from_pretrained(model_name,
    #config=config, 
    #ignore_mismatched_sizes=True
    )
    processor = CLIPProcessor.from_pretrained(model_name,
    #config=config
    )
    
    # Load image
 
    image = Image.open(io.BytesIO(image_path))
        
    # Process inputs
    inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Calculate similarity (cosine similarity)
    image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
    similarity = torch.matmul(text_embeds, image_embeds.T)[0][0].item()
    
    return similarity

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from PIL import Image
import requests
processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5')
text_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True).to("cuda")
text_model.eval()
async def calculate_visual_embeddings(image_path, text):


    image = Image.open(io.BytesIO(image_path))

    inputs = processor(image, return_tensors="pt").to("cuda")

    img_emb = vision_model(**inputs).last_hidden_state
    img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)




    encoded_input = tokenizer([text], padding=True, truncation=True, return_tensors='pt').to("cuda")

    with torch.no_grad():
        model_output = text_model(**encoded_input)

    text_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    text_embeddings = F.layer_norm(text_embeddings, normalized_shape=(text_embeddings.shape[1],))
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    align_score = torch.matmul(img_embeddings, text_embeddings.T)
    align_score = align_score.item()
    #print(f"Alignment score: {align_score}")
    return {"alignment_score": align_score}
