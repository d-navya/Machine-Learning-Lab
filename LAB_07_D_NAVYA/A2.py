import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tabulate import tabulate

# Load dataset
df = pd.read_csv('LAB_07_D_NAVYA\\GST - C_AST.csv')

# Feature-target split
X = df.drop('Final_Marks', axis=1)
y = df['Final_Marks']

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models and parameter grids
models = {
    'SVR': (SVR(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}),
    'DecisionTree': (DecisionTreeRegressor(), {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}),
    'RandomForest': (RandomForestRegressor(), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}),
    'CatBoost': (CatBoostRegressor(verbose=0), {'depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}),
    'AdaBoost': (AdaBoostRegressor(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}),
    'XGBoost': (XGBRegressor(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}),
    'MLP': (MLPRegressor(max_iter=1000), {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]})
}

results = []

# Hyperparameter tuning and evaluation
for name, (model, params) in models.items():
    random_search = RandomizedSearchCV(model, params, n_iter=5, cv=3, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    results.append([
        name,
        random_search.best_params_,
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred),
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)
    ])

# Display results in tabular format
print(tabulate(results, headers=["Model", "Best Params", "Train MSE", "Test MSE", "Train R2", "Test R2"], tablefmt="grid"))
