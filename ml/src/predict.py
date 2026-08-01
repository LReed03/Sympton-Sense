import pickle
import time
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = (
    BASE_DIR
    / "data"
    / "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
)

MODEL_DIRECTORY = BASE_DIR / "models"
MODEL_PATH = MODEL_DIRECTORY / "disease_model.pkl"


print("Loading dataset...")

df = pd.read_csv(CSV_PATH)

# Keep classes with enough examples for stratified splitting
class_counts = df["diseases"].value_counts()
valid_classes = class_counts[class_counts >= 5].index
df = df[df["diseases"].isin(valid_classes)].copy()

X = df.drop(columns=["diseases"]).astype("float32")
y = df["diseases"]

print(f"Rows: {len(df)}")
print(f"Features: {X.shape[1]}")
print(f"Diseases: {y.nunique()}")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    stratify=y,
)


model = LogisticRegression(
    solver="saga",
    max_iter=200,
    tol=0.01,
    random_state=0,
    n_jobs=-1,
    verbose=1,
)


print("\nTraining model...")
start_time = time.time()

model.fit(X_train, y_train)

elapsed_time = time.time() - start_time

print(f"\nTraining finished in {elapsed_time / 60:.2f} minutes.")


accuracy = model.score(X_test, y_test)

print(f"Test accuracy: {accuracy:.4f}")


y_pred = model.predict(X_test)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )
)


MODEL_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

with MODEL_PATH.open("wb") as model_file:
    pickle.dump(model, model_file)

print(f"\nModel saved to: {MODEL_PATH}")