# Real-Time ML Inference REST API & Capstone

This project packages the text-classification model from the NLP task into a small FastAPI microservice.

The service accepts a JSON payload at `/predict` and returns the predicted class, the confidence for that class, and the probability for every supported class.

## Project structure

```text
ml_inference_api/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── schemas.py
│   └── artifacts/
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── train_model.py
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

## Model

The model is a PyTorch multi-layer text classifier. Text is converted to TF-IDF features before being passed into the neural network.

The four classes are:

- `comp.graphics`
- `rec.sport.baseball`
- `rec.sport.hockey`
- `sci.space`

The training script is included so the repository can recreate the model artifacts instead of depending on a model file that only exists on one computer.

## API

### Health check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

### Prediction

```http
POST /predict
Content-Type: application/json
```

Request:

```json
{
  "text": "The hockey team played a great game."
}
```

Example response shape:

```json
{
  "prediction": "rec.sport.hockey",
  "confidence": 0.91,
  "probabilities": [
    {
      "label": "comp.graphics",
      "probability": 0.01
    },
    {
      "label": "rec.sport.baseball",
      "probability": 0.03
    },
    {
      "label": "rec.sport.hockey",
      "probability": 0.91
    },
    {
      "label": "sci.space",
      "probability": 0.05
    }
  ]
}
```

The exact probabilities will vary because they depend on the trained model.

## Running locally

Create and activate a virtual environment if desired, then install the pinned dependencies:

```bash
pip install -r requirements.txt
```

Train/export the model artifacts:

```bash
python train_model.py
```

Start the API:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI's interactive documentation is available at `/docs`.

## Running tests

After the model artifacts have been created:

```bash
pytest -q
```

The tests check:

- the health endpoint
- successful prediction response
- response schema fields
- invalid/missing text
- maximum text length
- HTTP status codes

## Docker

Build the image:

```bash
docker build -t nlp-inference-api .
```

Run it:

```bash
docker run --rm -p 8000:8000 nlp-inference-api
```

Then open:

```text
http://127.0.0.1:8000/docs
```

The Docker build runs `train_model.py`, so the image contains the generated model artifacts when the API starts.

## End-to-end architecture

```text
Client
  |
  | JSON text
  v
FastAPI /predict
  |
  v
Request validation (Pydantic)
  |
  v
TF-IDF Vectorizer
  |
  v
PyTorch Text Classifier
  |
  v
Softmax probabilities
  |
  v
JSON prediction response
```

## Notes

This repository is intentionally small and easy to follow. In a larger production system, model artifacts would normally be versioned and stored in a model registry or object storage, and the API would also include logging, authentication, monitoring, and a CI/CD pipeline.
