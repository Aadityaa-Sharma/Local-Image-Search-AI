# 🧠 Local Image Search AI (BLIP + CLIP)

Search **your own image folder** using real AI.
This tool automatically **generates captions + embeddings** for each image using BLIP & CLIP and lets you perform semantic search from the command line.

✔ Works **fully offline**
✔ No API keys
✔ Supports **natural language queries**
✔ Shows **image preview** for top matches
✔ Works even on **CPU**

---

## 📦 Features

| Feature                           | Status          |
| --------------------------------- | --------------- |
| Auto captioning (BLIP)            | ✔               |
| Image embeddings (CLIP)           | ✔               |
| Keyword extraction                | ✔               |
| Smart searching (CLIP + keywords) | ✔               |
| CLI interface                     | ✔               |
| Image preview popup               | ✔               |
| Web API version                   | ❌ (coming soon) |
| Vector DB (Faiss / Annoy)         | ❌ (future)      |

---

## 🧪 Demo

**Search example:**

```bash
python code/search_cli.py "a woman taking photo near tree"
```

**Output:**

```
Searching for: 'a woman taking photo near tree'

#1
  File    : test.jpg
  Caption : a woman taking a picture with a camera in front of a tree
  Keywords: woman, taking, picture, camera, front, tree
  CLIP    : 0.812
  KW      : 0.833
  Final   : 0.817
```

An image window shows the result using `matplotlib`.

---

## 📁 Project Structure

```
image-search-ai/
│
├── code/
│   ├── models.py          # Loads BLIP + CLIP models
│   ├── build_all.py       # Build index.json + index_with_keywords.json
│   ├── search_cli.py      # CLI search tool with image preview
│   └── process_image.py   # Single-image testing (optional)
│
├── data/
│   ├── images/            # Your image dataset goes here
│   ├── index.json         # Captions + embeddings
│   └── index_with_keywords.json
│
├── models/                # Downloaded model cache (ignored in git)
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Installation

```bash
# clone repo
git clone <repo_link>
cd image-search-ai

# create venv
python -m venv venv
source venv/bin/activate  # mac / linux
venv\Scripts\activate     # windows

# install dependencies
pip install -r requirements.txt
```

---

## 🖼️ Add Images

Place your images in:

```
data/images/
```

---

## 🏗️ Build the Search Index

```bash
python code/build_all.py
```

This generates or overwrites the :

| File                            | Purpose                            |
| ------------------------------- | ---------------------------------- |
| `data/index.json`               | Caption + embedding for each image |
| `data/index_with_keywords.json` | + keywords extracted from caption  |

---

## 🔍 Search from CLI (Interactive Mode)

```bash
python code/search_cli.py
# then type queries:
> a mountain lake house
> a man riding horse
> quit
```

Or one-shot mode:

```bash
python code/search_cli.py "a brown horse in field"
```

---

## 📦 Requirements

```
torch
torchvision
transformers
pillow
numpy
matplotlib
tqdm
```

Install via:

```bash
pip install -r requirements.txt
```

---

## 🚀 Roadmap / Future Upgrades

| Feature                 | Status |
| ----------------------- | ------ |
| FastAPI web API         | 🔜     |
| Frontend UI (Tailwind)  | 🔜     |
| Vector DB (FAISS/Annoy) | 🔜     |
| Mobile version          | 🔜     |
| GPU acceleration        | 🔜     |

---

## 🙌 Credits

* **BLIP** by Salesforce
* **CLIP** by OpenAI

---

**⭐ If you like it — star the repo and share your results!**

---
