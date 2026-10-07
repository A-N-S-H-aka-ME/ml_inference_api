from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class PredictionItem(BaseModel):
    label: str
    probability: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    prediction: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: list[PredictionItem]
