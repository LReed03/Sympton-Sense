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
        columns=model.feature_names_in_,
    )
    for symptom in symptoms:
        if symptom in input_data.columns:
            input_data[symptom] = 1
    prediction = model.predict(input_data)

    return prediction.tolist()