
import os
import sys
import json
import re

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models import get_text_embedding, caption_to_keywords

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
INDEX_WITH_KW = os.path.join(PROJECT_ROOT, "data", "index_with_keywords.json")
INDEX_PLAIN = os.path.join(PROJECT_ROOT, "data", "index.json")  


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    Value is between -1 and 1 (1 = most similar).
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def load_index():
    """
    Try loading index_with_keywords.json first.
    If not present, fall back to index.json (without keywords).
    """
    if os.path.exists(INDEX_WITH_KW):
        path = INDEX_WITH_KW
    else:
        path = INDEX_PLAIN

    with open(path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from {path}")
    return data


def compute_scores(query: str, index_entries: list[dict], top_k: int = 5):
    """
    For a given text query:
    - compute CLIP embedding
    - compute cosine similarity with each image embedding
    - compute keyword overlap score with BLIP keywords (if present)
    - combine: 0.8 * clip_score + 0.2 * keyword_score
    - return top_k results sorted by final score
    """
    query_embed = get_text_embedding(query)  # numpy vector

    
    query_keywords = caption_to_keywords(query)
    query_kw_set = set(query_keywords)

    results: list[dict] = []

    for item in index_entries:
        
        emb = np.array(item["embedding"], dtype=np.float32)

        
        clip_score = cosine_similarity(query_embed, emb)

        
        item_keywords = item.get("keywords", [])
        item_kw_set = set(item_keywords)

        if query_kw_set and item_kw_set:
            overlap = len(query_kw_set & item_kw_set)
            keyword_score = overlap / len(query_kw_set)
        else:
            keyword_score = 0.0

        
        final_score = 0.8 * clip_score + 0.2 * keyword_score

        results.append({
            "filename": item["filename"],
            "caption": item.get("caption", ""),
            "keywords": item_keywords,
            "clip_score": clip_score,
            "keyword_score": keyword_score,
            "final_score": final_score
        })

    
    results.sort(key=lambda x: x["final_score"], reverse=True)

    return results[:top_k]


def main():
    
    index_entries = load_index()

    
    initial_query = " ".join(sys.argv[1:]) if len(sys.argv) >= 2 else None

    while True:
        if initial_query is not None:
            query = initial_query.strip()
            initial_query = None  
        else:
            query = input("\nEnter search query (or 'q' to quit): ").strip()

        if not query:
            print("Empty query. Try again or type 'q' to quit.")
            continue

        if query.lower() in {"q", "quit", "exit"}:
            print("Exiting search.")
            break

        print(f"\nSearching for: {query!r}\n")

        top_results = compute_scores(query, index_entries, top_k=5)

        if not top_results:
            print("No results found.")
            continue

        for rank, r in enumerate(top_results, start=1):
            print(f"#{rank}")
            print(f"  File    : {r['filename']}")
            print(f"  Caption : {r['caption']}")
            if r["keywords"]:
                print(f"  Keywords: {', '.join(r['keywords'])}")
            print(f"  CLIP    : {r['clip_score']:.3f}")
            print(f"  KW      : {r['keyword_score']:.3f}")
            print(f"  Final   : {r['final_score']:.3f}")
            print()

            
            img_path = os.path.join(PROJECT_ROOT, "data", "images", r["filename"])
            if os.path.exists(img_path):
                image = Image.open(img_path)
                plt.imshow(image)
                plt.title(f"#{rank}: {r['filename']}  score={r['final_score']:.3f}")
                plt.axis("off")
                plt.show()
            else:
                print("  [Image file not found on disk]")


if __name__ == "__main__":
    main()
