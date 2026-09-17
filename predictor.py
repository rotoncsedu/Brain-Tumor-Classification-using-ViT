import io
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

CHECKPOINT = "google/vit-base-patch16-224"

# Resolve artifacts relative to this file, not the process's current working
# directory, so the API works no matter where uvicorn is launched from.
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
WEIGHTS_PATH = ARTIFACTS_DIR / "vit_brain_tumor.pt"

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {CLASS_NAMES_PATH}. Make sure the 'artifacts' folder "
        "(class_names.json + vit_brain_tumor.pt) sits next to predictor.py."
    )
if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {WEIGHTS_PATH}. Make sure the 'artifacts' folder "
        "(class_names.json + vit_brain_tumor.pt) sits next to predictor.py."
    )

with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_processor = ViTImageProcessor.from_pretrained(CHECKPOINT)

model = ViTForImageClassification.from_pretrained(
    CHECKPOINT,
    num_labels=len(CLASS_NAMES),
    ignore_mismatched_sizes=True,
)

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


def predict(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred_idx = probs.argmax().item()

    return {
        "label": CLASS_NAMES[pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
    }