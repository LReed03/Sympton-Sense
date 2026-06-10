import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


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


# Split the dataset into training and testing sets, ensuring that the class distribution is maintained in both sets using stratification.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    stratify=y
)

# Create a machine learning pipeline that includes a StandardScaler for feature scaling and a Logistic Regression classifier. This pipeline will be used to perform hyperparameter tuning with GridSearchCV.
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=0))
])

# Define a parameter grid for both Logistic Regression and Random Forest to perform hyperparameter tuning using GridSearchCV.
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

# Fit the GridSearchCV object to the training data to find the best hyperparameters for the model. After fitting, print the best parameters, cross-validation scores, and test-set score. Finally, save the best model to a file using pickle.
grid.fit(X_train, y_train)

print("Best params:")
print(grid.best_params_)

print("Best cross-validation train score: {:.2f}".format(
    grid.cv_results_["mean_train_score"][grid.best_index_]
))

print("Best cross-validation test score: {:.2f}".format(grid.best_score_))

print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))

y_pred = grid.predict(X_test)

print(classification_report(y_test, y_pred))

# Create the "models" directory if it doesn't exist, and save the best model from GridSearchCV to a file named "disease_model.pkl" using pickle. This allows the model to be loaded later for making predictions without needing to retrain it.
models_dir = BASE_DIR / "models"
models_dir.mkdir(exist_ok=True)

model_path = models_dir / "disease_model.pkl"

with open(model_path, "wb") as f:
    pickle.dump(grid.best_estimator_, f)

print(f"Model saved to: {model_path}")