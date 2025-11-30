
import os
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel
)

device = "cuda" if torch.cuda.is_available() else "cpu"


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
BLIP_CACHE_DIR = os.path.join(PROJECT_ROOT, "models", "blip")
CLIP_CACHE_DIR = os.path.join(PROJECT_ROOT, "models", "clip")


BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-large"
blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME, cache_dir=BLIP_CACHE_DIR)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME, cache_dir=BLIP_CACHE_DIR).to(device)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, cache_dir=CLIP_CACHE_DIR)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, cache_dir=CLIP_CACHE_DIR).to(device)

def generate_caption(image):
    inputs = blip_processor(image, return_tensors="pt").to(device)
    output = blip_model.generate(**inputs, max_new_tokens=60)
    return blip_processor.decode(output[0], skip_special_tokens=True)

def get_image_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embeds = clip_model.get_image_features(**inputs)
    return embeds[0].cpu().numpy()

def get_text_embedding(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embeds = clip_model.get_text_features(**inputs)
    return embeds[0].cpu().numpy()
