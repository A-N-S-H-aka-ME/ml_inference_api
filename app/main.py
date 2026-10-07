from fastapi import FastAPI, HTTPException

from app.model import Predictor
from app.schemas import PredictRequest, PredictResponse


app = FastAPI(
    title="NLP Text Classifier API",
    description="A small production-style FastAPI service for text classification.",
    version="1.0.0",
)

_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        try:
            _predictor = Predictor()
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Model artifacts are missing. Run train_model.py first."
            ) from exc
    return _predictor


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")

    return get_predictor().predict(text)
