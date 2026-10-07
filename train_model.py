from pathlib import Path

import joblib
import torch
import torch.nn as nn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from app.model import TextClassifier


ARTIFACT_DIR = Path("app/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.space",
    "comp.graphics",
]


def main():
    print("Downloading/loading the training data...")

    train_data = fetch_20newsgroups(
        subset="train",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
    )

    # The same TF-IDF idea used in the notebook from Task 05.
    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words="english",
    )

    X = vectorizer.fit_transform(train_data.data)
    y = train_data.target

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
    X_val = torch.tensor(X_val.toarray(), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=64,
        shuffle=False,
    )

    model = TextClassifier(X_train.shape[1], len(CATEGORIES))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    patience = 3
    wait = 0

    print("Training the classifier...")

    for epoch in range(12):
        model.train()
        train_loss = 0.0

        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                val_loss += criterion(outputs, labels).item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch + 1:02d}/12 | "
            f"Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), ARTIFACT_DIR / "model.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    joblib.dump(vectorizer, ARTIFACT_DIR / "vectorizer.joblib")
    joblib.dump(
        {
            "labels": train_data.target_names,
            "input_size": X_train.shape[1],
        },
        ARTIFACT_DIR / "metadata.joblib",
    )

    print("Model artifacts saved in app/artifacts/")


if __name__ == "__main__":
    main()
