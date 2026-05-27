import pandas as pd
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "models" / "disease_model.pkl"
csv_path = BASE_DIR / "data" / "Final_Augmented_dataset_Diseases_and_Symptoms.csv"

with open(model_path, "rb") as f:
    model = pickle.load(f)

df = pd.read_csv(csv_path)
feature_names = df.drop(columns=["diseases"]).columns.tolist()


def predict_disease(selected_symptoms):
    user_symptoms = {symptom : 0 for symptom in feature_names}

    for symptom in selected_symptoms:
        if symptom in user_symptoms:
            user_symptoms[symptom] = 1
    input_data = pd.DataFrame([user_symptoms])

    predicted_disease = model.predict(input_data)[0]
    return predicted_disease