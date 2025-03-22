import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tabulate import tabulate

# Load dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop('Final_Marks', axis=1)
    y = df['Final_Marks']
    return X, y

# Preprocess data
def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Define models and hyperparameters
def get_models():
    return {
        'SVR': (SVR(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}),
        'DecisionTree': (DecisionTreeRegressor(), {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}),
        'RandomForest': (RandomForestRegressor(), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}),
        'AdaBoost': (AdaBoostRegressor(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}),
        'XGBoost': (XGBRegressor(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}),
        'MLP': (MLPRegressor(max_iter=1000), {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]})
    }

# Train and evaluate models
def train_evaluate_models(models, X_train, X_test, y_train, y_test):
    results = []
    performance = {}
    for name, (model, params) in models.items():
        random_search = RandomizedSearchCV(model, params, n_iter=5, cv=3, random_state=42, n_jobs=-1)
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        results.append([name, random_search.best_params_, train_mse, test_mse, train_r2, test_r2])
        performance[name] = {'Train MSE': train_mse, 'Test MSE': test_mse, 'Train R2': train_r2, 'Test R2': test_r2}
    return results, performance

# Display results
def display_results(results):
    print(tabulate(results, headers=["Model", "Best Params", "Train MSE", "Test MSE", "Train R2", "Test R2"], tablefmt="grid"))

# Plot performance
def plot_performance(performance):
    models = list(performance.keys())
    test_mse = [performance[m]['Test MSE'] for m in models]
    test_r2 = [performance[m]['Test R2'] for m in models]
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.bar(models, test_mse, color='b', alpha=0.6, label='Test MSE')
    ax2.plot(models, test_r2, color='r', marker='o', label='Test R2')
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Test MSE', color='b')
    ax2.set_ylabel('Test R2', color='r')
    ax1.set_xticklabels(models, rotation=45)
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('Model Performance Comparison')
    plt.show()

# Main execution
def main():
    X, y = load_data('LAB_07_D_NAVYA\\GST - C_AST.csv')
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    models = get_models()
    results, performance = train_evaluate_models(models, X_train, X_test, y_train, y_test)
    display_results(results)
    plot_performance(performance)

if __name__ == "__main__":
    main()