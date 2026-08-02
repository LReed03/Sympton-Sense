import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = (
    BASE_DIR
    / "data"
    / "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
)

MODEL_DIRECTORY = BASE_DIR / "models"
MODEL_PATH = MODEL_DIRECTORY / "disease_model.pkl"


# Load dataset
df = pd.read_csv(CSV_PATH)

# Keep diseases with enough samples for stratification and cross-validation
class_counts = df["diseases"].value_counts()
valid_classes = class_counts[class_counts >= 5].index
df = df[df["diseases"].isin(valid_classes)].copy()

# Separate symptoms and disease labels
X = df.drop(columns=["diseases"])
y = df["diseases"]

print(f"Samples: {len(df)}")
print(f"Symptoms: {X.shape[1]}")
print(f"Diseases: {y.nunique()}")


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    stratify=y,
)


# Model pipeline
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=0,
                solver="saga",
            ),
        ),
    ]
)


# Models to compare
param_grid = [
    {
        "classifier": [
            LogisticRegression(
                max_iter=1000,
                random_state=0,
                solver="saga",
            )
        ],
        "classifier__C": [0.1, 1, 10],
    },
    {
        "scaler": ["passthrough"],
        "classifier": [
            RandomForestClassifier(
                random_state=0,
                n_jobs=1,
            )
        ],
        "classifier__n_estimators": [100],
        "classifier__max_depth": [10, None],
    },
]


# Train and find the best model
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=1,
    verbose=2,
)

print("\nBeginning model training...")

grid.fit(X_train, y_train)

best_model = grid.best_estimator_


# Evaluate model
y_pred = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)

test_accuracy = accuracy_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)

top_3_accuracy = top_k_accuracy_score(
    y_test,
    probabilities,
    k=3,
    labels=best_model.classes_,
)

top_5_accuracy = top_k_accuracy_score(
    y_test,
    probabilities,
    k=5,
    labels=best_model.classes_,
)


print("\nModel results:")
print(f"Best model: {grid.best_params_}")
print(f"Cross-validation accuracy: {grid.best_score_:.2%}")
print(f"Top-1 test accuracy: {test_accuracy:.2%}")
print(f"Top-3 accuracy: {top_3_accuracy:.2%}")
print(f"Top-5 accuracy: {top_5_accuracy:.2%}")
print(f"Macro F1 score: {macro_f1:.4f}")


# Save model
MODEL_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

with MODEL_PATH.open("wb") as model_file:
    pickle.dump(best_model, model_file)

print(f"\nModel saved to: {MODEL_PATH}")