from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


MODEL_PATH = Path(__file__).with_name("dropout_model.pkl")
FEATURE_NAMES = [
    "gpa",
    "attendance",
    "study_hours",
    "family_income",
    "extracurricular",
    "previous_failures",
    "age",
    "internet_access",
]


def _build_dataset() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=400,
        n_features=len(FEATURE_NAMES),
        n_informative=6,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.7, 0.3],
        random_state=42,
    )
    return X, y


def train_model() -> dict[str, Any]:
    X, y = _build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    metadata = {
        "model_type": "RandomForestClassifier",
        "accuracy": float(accuracy),
        "feature_names": FEATURE_NAMES,
    }
    _save_model(model, metadata)
    return {"model": model, **metadata}


def _save_model(model: RandomForestClassifier, metadata: dict[str, Any]) -> None:
    with MODEL_PATH.open("wb") as handle:
        pickle.dump({"metadata": metadata, "model": model}, handle)


def load_model() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        return train_model()

    with MODEL_PATH.open("rb") as handle:
        payload = pickle.load(handle)

    return {"model": payload["model"], **payload["metadata"]}


def predict_dropout(student: dict[str, Any], model: dict[str, Any] | None = None) -> dict[str, Any]:
    if model is None:
        model = load_model()

    features = np.array([student[name] for name in model["feature_names"]], dtype=float).reshape(1, -1)
    probability = float(model["model"].predict_proba(features)[0, 1])
    label = "dropout" if probability >= 0.5 else "not_dropout"
    return {"label": label, "probability": probability}


if __name__ == "__main__":
    model = train_model()
    sample = {
        "gpa": 2.1,
        "attendance": 60,
        "study_hours": 4,
        "family_income": 25000,
        "extracurricular": 0,
        "previous_failures": 2,
        "age": 19,
        "internet_access": 1,
    }
    result = predict_dropout(sample, model)
    print({"model_accuracy": model["accuracy"], **result})
