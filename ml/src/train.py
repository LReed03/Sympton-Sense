import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
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


# Load the dataset
df = pd.read_csv(CSV_PATH)

# Keep diseases with enough samples for stratification and cross-validation
class_counts = df["diseases"].value_counts()
valid_classes = class_counts[class_counts >= 5].index

df = df[df["diseases"].isin(valid_classes)].copy()

# Separate symptoms and disease labels
X = df.drop(columns=["diseases"])
y = df["diseases"]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Number of diseases: {y.nunique()}")
print("Missing X values:", X.isnull().sum().sum())
print("Missing y values:", y.isnull().sum())


# Create train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    stratify=y,
)


pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=0,
                solver="saga",
                l1_ratio=0,
            ),
        ),
    ]
)


# Define a parameter grid for both Logistic Regression and Random Forest to perform hyperparameter tuning using GridSearchCV.
param_grid = [
    {
        "classifier": [
            LogisticRegression(
                max_iter=1000,
                random_state=0,
                solver="saga",
                l1_ratio=0,
            )
        ],
        "classifier__C": [0.1, 1, 10],
    },
    {
        "classifier": [
            RandomForestClassifier(
                random_state=0,
                n_jobs=1,
            )
        ],
        "classifier__n_estimators": [50],
        "classifier__max_depth": [10],
    },
]


grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    return_train_score=True,
    n_jobs=1,
    pre_dispatch=1,
    verbose=2,
)


print("\nBeginning model training...")

grid.fit(X_train, y_train)


print("\nBest parameters:")
print(grid.best_params_)

print(
    "Best cross-validation train score: "
    f"{grid.cv_results_['mean_train_score'][grid.best_index_]:.4f}"
)

print(
    "Best cross-validation validation score: "
    f"{grid.best_score_:.4f}"
)

print(
    "Test-set score: "
    f"{grid.score(X_test, y_test):.4f}"
)


y_pred = grid.predict(X_test)

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
    pickle.dump(
        grid.best_estimator_,
        model_file,
    )

print(f"\nModel saved to: {MODEL_PATH}")