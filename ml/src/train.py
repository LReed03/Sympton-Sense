import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")


df = pd.read_csv('../data/Final_Augmented_dataset_Diseases_and_Symptoms.csv')


class_counts = df["diseases"].value_counts()
valid_classes = class_counts[class_counts >= 5].index
df = df[df["diseases"].isin(valid_classes)]

X = df.drop(columns=["diseases"])
y = df["diseases"]
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")



print(X.isnull().sum())
print('-----------------------------')
print(y.isnull().sum())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
split_X_train, X_val, split_y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state = 0, stratify=y_train)


scaler = StandardScaler()

pipe = Pipeline([
    ("scaler", scaler),
    ("classifier", LogisticRegression(max_iter=1000, random_state=0))
])


param_grid = [
    {
        'classifier': [LogisticRegression(max_iter=1000, random_state=0, solver='liblinear')],
        'classifier__C': [0.1, 1, 10],
        'classifier__penalty': ['l1', 'l2']
    },
    {
        'classifier': [RandomForestClassifier(random_state=0)],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, None],
        'classifier__max_features': ['sqrt']
    }
]

grid = GridSearchCV(pipe, param_grid, cv=5, return_train_score=True, n_jobs=-1)
grid.fit(split_X_train, split_y_train)

print("Best params:\n{}\n".format(grid.best_params_))
print("Best cross-validation train score: {:.2f}".format(grid.cv_results_['mean_train_score'][grid.best_index_]))
print("Best cross-validation test score: {:.2f}".format(grid.best_score_))
print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))