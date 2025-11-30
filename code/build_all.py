
import os
import json
import re
from typing import List

from PIL import Image
from models import generate_caption, get_image_embedding
from tqdm import tqdm  # progress bar for long loops


# ---------------- PATH SETUP ----------------

# PROJECT_ROOT = parent folder of /code
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Folder containing your images
IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "images")

# Basic index (caption + embedding)
INDEX_JSON = os.path.join(PROJECT_ROOT, "data", "index.json")

# Extended index (caption + embedding + keywords)
INDEX_WITH_KW_JSON = os.path.join(PROJECT_ROOT, "data", "index_with_keywords.json")


# ------------- KEYWORD EXTRACTION -------------

def caption_to_keywords(caption: str) -> List[str]:
    """
    Convert a sentence-like caption into a list of simple keywords.
    Steps:
    1. lowercase the text
    2. remove punctuation
    3. split into words
    4. remove very common small words (stopwords)
    5. remove duplicates while keeping order
    """
    caption = caption.lower()

    # Keep only letters, digits and spaces → everything else becomes space
    caption = re.sub(r"[^a-z0-9 ]+", " ", caption)

    # Split into individual words
    words = caption.split()

    # Very common words that don't help searching much
    stopwords = {
        "a", "an", "the", "and", "or", "of", "in", "on", "with", "for", "to",
        "from", "by", "at", "near", "over", "under", "is", "are", "this",
        "that", "these", "those", "into", "onto", "as", "it", "its", "his",
        "her", "their", "our", "your", "you", "i", "we", "they"
    }

    filtered = [w for w in words if w not in stopwords and len(w) > 2]

    # Remove duplicates but keep original order
    seen = set()
    keywords: List[str] = []
    for w in filtered:
        if w not in seen:
            seen.add(w)
            keywords.append(w)

    return keywords


# ------------- MAIN PIPELINE FUNCTION -------------

def build_all():
    """
    Full pipeline:
    - Go through all images in IMAGE_DIR
    - For each image:
        * open image
        * generate caption using BLIP
        * generate embedding using CLIP
    - Save basic index.json (without keywords)
    - Add keywords from captions
    - Save index_with_keywords.json
    """

    if not os.path.isdir(IMAGE_DIR):
        print(f"Image folder not found: {IMAGE_DIR}")
        return

    entries = []

    # Get list of files once, then wrap it with tqdm for a progress bar
    file_list = sorted(os.listdir(IMAGE_DIR))

    # Loop over all files in the image directory with a progress bar
    for filename in tqdm(file_list, desc="Processing images", unit="img"):
        # Process only typical image extensions
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        image_path = os.path.join(IMAGE_DIR, filename)
        # print(f"Processing image: {filename} ...")  # tqdm already shows progress

        # Open image with Pillow
        image = Image.open(image_path).convert("RGB")

        # ---- BLIP caption ----
        caption = generate_caption(image)

        # ---- CLIP embedding ----
        embedding = get_image_embedding(image)  # numpy array

        # Store basic info in Python dict
        entries.append({
            "filename": filename,
            "caption": caption,
            "embedding": embedding.tolist()   # convert numpy array → normal list for JSON
        })

    # Ensure data folder exists
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)

    # --------- SAVE BASIC INDEX (NO KEYWORDS) ---------
    with open(INDEX_JSON, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"\nSaved basic index to: {INDEX_JSON}")

    # --------- ADD KEYWORDS & SAVE EXTENDED INDEX ---------
    for item in entries:
        kw = caption_to_keywords(item.get("caption", ""))
        item["keywords"] = kw

    with open(INDEX_WITH_KW_JSON, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Saved index with keywords to: {INDEX_WITH_KW_JSON}")

    print(f"\nDone. Total images processed: {len(entries)}")


if __name__ == "__main__":
    build_all()
