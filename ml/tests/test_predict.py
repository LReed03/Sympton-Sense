import pytest
from src.predict import predict_disease

def test_prediction_returns_results():
    symptoms = ["cough", "fever", "fatigue"]

    results = predict_disease(symptoms)
    print(f"Results for symptoms {symptoms}: {results}")

    assert results is not None
    assert len(results) > 0

def test_single_symptom():
    results = predict_disease(["cough"])

    print(f"Results for single symptom: {results}")
    assert isinstance(results, list)
    assert len(results) > 0
