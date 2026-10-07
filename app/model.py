from pathlib import Path

import joblib
import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class Predictor:
    def __init__(self, artifact_dir: str = "app/artifacts"):
        artifact_path = Path(artifact_dir)

        self.vectorizer = joblib.load(artifact_path / "vectorizer.joblib")
        metadata = joblib.load(artifact_path / "metadata.joblib")

        self.labels = metadata["labels"]
        input_size = metadata["input_size"]
        num_classes = len(self.labels)

        self.model = TextClassifier(input_size, num_classes)
        self.model.load_state_dict(
            torch.load(
                artifact_path / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        self.model.eval()

    def predict(self, text: str):
        features = self.vectorizer.transform([text]).toarray()
        tensor = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        predicted_index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_index].item())

        items = [
            {
                "label": self.labels[i],
                "probability": float(probabilities[i].item()),
            }
            for i in range(len(self.labels))
        ]

        return {
            "prediction": self.labels[predicted_index],
            "confidence": confidence,
            "probabilities": items,
        }
