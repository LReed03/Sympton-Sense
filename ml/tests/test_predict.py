import pytest
from src.predict import predict_disease

def test_prediction_returns_results():
    symptoms = ["cough", "fever", "fatigue"]

    results = predict_disease(symptoms)

    assert results is not None
    assert len(results) > 0