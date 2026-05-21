import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV



df = pd.read_csv('../data/Final_Augmented_dataset_Diseases_and_Symptoms.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
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
    'classifier': [LogisticRegression(max_iter=1000, random_state=0)],
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__fit_intercept': [True, False]
    },
    {
    'classifier': [KNeighborsClassifier()],
    'classifier__n_neighbors': [3, 5, 7, 9],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
    },
    {
    'classifier': [RandomForestClassifier(random_state=0)],
    'classifier__n_estimators': [300, 400, 500],
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [10, 20, 50],
    'classifier__min_samples_leaf': [5, 10, 20],
    'classifier__max_features': ['sqrt', 'log2'],
    }
]

grid = GridSearchCV(pipe, param_grid, cv=5, return_train_score=True, n_jobs=-1)
grid.fit(split_X_train, split_y_train)

print("Best params:\n{}\n".format(grid.best_params_))
print("Best cross-validation train score: {:.2f}".format(grid.cv_results_['mean_train_score'][grid.best_index_]))
print("Best cross-validation test score: {:.2f}".format(grid.best_score_))
print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))