from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import predictor

app = FastAPI(
    title="Brain Tumor Detection Inference API",
    description="Serves a ViT model fine-tuned to detect brain tumors in MRI scans.",
    version="1.0.0",
)

# Allow the browser GUI (opened as a local file, or served from a different
# port) to call this API. Tighten allow_origins to a specific origin before
# deploying this publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionOutput(BaseModel):
    label: str
    confidence: float


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "ViT model fine-tuned to detect brain tumors in MRI scans",
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file, got content-type '{file.content_type}'.",
        )

    try:
        image_bytes = await file.read()
        result = predictor.predict(image_bytes)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))