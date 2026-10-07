from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_schema_and_status():
    response = client.post(
        "/predict",
        json={"text": "The hockey team played a great game."},
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "confidence" in data
    assert "probabilities" in data

    assert isinstance(data["prediction"], str)
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 4


def test_empty_text_is_rejected():
    response = client.post("/predict", json={"text": "   "})

    assert response.status_code == 422


def test_missing_text_is_rejected():
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_text_too_long_is_rejected():
    response = client.post("/predict", json={"text": "a" * 5001})

    assert response.status_code == 422
