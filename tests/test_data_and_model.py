
import json
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "bank_marketing_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"


def test_model_file_exists():
    assert MODEL_PATH.exists(), "Trained model file is missing"


def test_metrics_file_exists():
    assert METRICS_PATH.exists(), "Metrics file is missing"


def test_model_prediction():
    model = joblib.load(MODEL_PATH)

    customer = pd.DataFrame([{
        "age": 42,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "balance": 1500,
        "housing": "no",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
    }])

    prediction = model.predict(customer)
    probability = model.predict_proba(customer)

    assert prediction[0] in [0, 1]
    assert 0 <= probability[0][1] <= 1


def test_metrics_values():
    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)

    assert "accuracy" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
