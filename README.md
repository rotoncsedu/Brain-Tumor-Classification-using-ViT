<div align="center">

# 🧠 Brain Tumor MRI Classifier

**A REST API serving a fine-tuned Vision Transformer (ViT) model that classifies brain MRI scans as glioma, meningioma, pituitary tumor, or no tumor.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-⚡-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-ViT-FFD21E)](https://huggingface.co/docs/transformers)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Model](#-model)
- [API Endpoints](#-api-endpoints)
- [Try It Out](#-try-it-out)
- [Running Locally](#-running-locally)
- [Project Structure](#-project-structure)

---

## 🧭 Overview

This project fine-tunes a **Vision Transformer (ViT)** on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) to classify brain MRI scans into four categories, then serves it behind a FastAPI application. Upload an MRI image, and the API returns a predicted class with a confidence score.

## 🧠 Model

| | |
|---|---|
| **Architecture** | `google/vit-base-patch16-224` (fine-tuned, `ignore_mismatched_sizes=True` for 4 output classes) |
| **Dataset** | [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle) |
| **Task** | 4-class image classification |
| **Output** | 4 logits → `softmax` → predicted class + confidence |
| **Epochs** | 6 |
| **Test Accuracy** | **94.44%** |

### Classes

| Class | Description |
|-------|-------------|
| `glioma` | Tumor arising from glial (supportive) cells of the brain or spinal cord |
| `meningioma` | Tumor arising from the meninges, the membranes surrounding the brain and spinal cord |
| `pituitary` | Tumor arising in the pituitary gland at the base of the brain |
| `notumor` | No tumor present in the scan |

The raw probabilities are converted into a clean response:

- 🏷️ **`label`** — the predicted class name
- 📊 **`confidence`** — the model's probability for the predicted class

### How inference works

At startup, the API loads two artifacts from `artifacts/`:

| File | Purpose |
|---|---|
| `vit_brain_tumor.pt` | Fine-tuned ViT weights (state_dict) |
| `class_names.json` | Ordered list of class labels |

```
MRI image  →  resize + normalize (ViTImageProcessor)  →  ViT  →  softmax  →  label + confidence
```

1. **Preprocess** — `ViTImageProcessor` resizes the image to 224×224 and normalizes it
2. **Predict** — the pixel values pass through the fine-tuned ViT to produce 4 logits
3. **Softmax** — converted into a probability distribution over the 4 classes

---

## 🔌 API Endpoints

| Method | Endpoint | Input | Output |
|---|---|---|---|
| `GET` | `/health` | none | `{"status": "ok", "model": "ViT model fine-tuned to detect brain tumors in MRI scans"}` |
| `POST` | `/predict` | `multipart/form-data` with a single field `file` (JPEG/PNG MRI image) | `{"label": str, "confidence": float}` |

<details>
<summary><strong>▶ Example request</strong></summary>

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@sample_scan.jpg"
```

</details>

<details>
<summary><strong>▶ Example response</strong></summary>

```json
{
  "label": "meningioma",
  "confidence": 0.9989
}
```

</details>

<details>
<summary><strong>▶ Error responses</strong></summary>

| Status | Meaning |
|--------|---------|
| `200` | Prediction succeeded |
| `400` | Uploaded file is not an image |
| `500` | Unexpected error during inference |

</details>

---

## 🖥️ Try It Out

### Swagger UI

FastAPI's auto-generated interactive docs at `/docs` let you send requests and inspect the JSON response directly in the browser.

<div align="center">

<img src="screenshots/swagger.png" alt="Swagger UI - /docs" width="480">

</div>

### Standalone UI

For a friendlier experience, `index.html` provides a standalone page that uploads a scan and displays the prediction as a live readout.

<div align="center">

<img src="screenshots/gui1.png" alt="Standalone UI - upload" width="480">
<img src="screenshots/gui2.png" alt="Standalone UI - result" width="480">

</div>

---

## 🚀 Running Locally

**1. Clone the repository**
```bash
git clone https://github.com/rotoncsedu/Brain-Tumor-Classification-using-ViT
cd Brain-Tumor-Classification-using-ViT
```

**2. Place the trained artifacts**

Copy your `vit_brain_tumor.pt` and `class_names.json` (produced by the training notebook) into `artifacts/`.

**3. Create and activate a virtual environment** *(optional but recommended)*
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

> ⚠️ First run needs internet access once — the ViT backbone (`google/vit-base-patch16-224`) is fetched from Hugging Face before the fine-tuned weights are loaded on top of it.

**5. Run the API**
```bash
fastapi dev main.py
```
> If `fastapi dev` isn't available, install the CLI extras (`pip install "fastapi[standard]"`), or run with `uvicorn main:app --reload` instead.

The server starts at `http://localhost:8000`.

**6. Try it out**

| Where | Link |
|---|---|
| 📘 Interactive docs | [http://localhost:8000/docs](http://localhost:8000/docs) |
| ❤️ Health check | [http://localhost:8000/health](http://localhost:8000/health) |
| 🖼️ Standalone UI | Open `gui.html` in a browser |

---

## 📁 Project Structure

```
Brain-Tumor-Classification-using-ViT/
├── main.py                    # FastAPI app and route definitions
├── predictor.py                 # Loads model and runs inference
├── index.html                       # Standalone UI
├── requirements.txt
├── screenshots/
│   ├── swagger.png                # Swagger UI (/docs) screenshot
│   ├── gui1.png                    # Standalone UI - upload
│   └── gui2.png                    # Standalone UI - result
├── artifacts/
│   ├── vit_brain_tumor.pt          # Fine-tuned model weights
│   └── class_names.json             # Ordered list of class labels
└── training/
    └── vit_finetuning.ipynb          # Kaggle training notebook
```

---

## 🛠️ Technologies Used

- Python
- PyTorch, Hugging Face Transformers
- FastAPI
- Pillow

## ⚠️ Disclaimer

This model is for educational/research purposes only and is **not a certified diagnostic tool**. Predictions should never be used as a substitute for professional medical evaluation.