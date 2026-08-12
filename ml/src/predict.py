import pickle
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "disease_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully.")

def predict_disease(symptoms):
    input_data = pd.DataFrame(
        [[0] * len(model.feature_names_in_)],
        columns=model.feature_names_in_
    )

    for symptom in symptoms:
        if symptom in input_data.columns:
            input_data.loc[0, symptom] = 1

    probabilities = model.predict_proba(input_data)[0]

    top_indices = probabilities.argsort()[-5:][::-1]

    results = []

    for index in top_indices:
        results.append({
            "disease": model.classes_[index],
            "probability": float(probabilities[index])
        })

    return results