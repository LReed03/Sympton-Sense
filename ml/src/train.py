import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
csv_path = BASE_DIR / "data" / "Final_Augmented_dataset_Diseases_and_Symptoms.csv"

df = pd.read_csv(csv_path)

class_counts = df["diseases"].value_counts()
valid_classes = class_counts[class_counts >= 5].index
df = df[df["diseases"].isin(valid_classes)]

X = df.drop(columns=["diseases"])
y = df["diseases"]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print("Missing X values:", X.isnull().sum().sum())
print("Missing y values:", y.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=0))
])

param_grid = [
    {
        "classifier": [
            LogisticRegression(
                max_iter=1000,
                random_state=0,
                solver="saga"
            )
        ],
        "classifier__C": [0.1, 1, 10],
        "classifier__penalty": ["l2"]
    },
    {
        "classifier": [
            RandomForestClassifier(
                random_state=0,
                n_jobs=-1
            )
        ],
        "classifier__n_estimators": [50],
        "classifier__max_depth": [10]
    }
]

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=3,
    return_train_score=True,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best params:")
print(grid.best_params_)

print("Best cross-validation train score: {:.2f}".format(
    grid.cv_results_["mean_train_score"][grid.best_index_]
))

print("Best cross-validation test score: {:.2f}".format(grid.best_score_))

print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))