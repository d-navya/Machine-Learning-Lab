import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Function to load data from the Excel file
def load_data(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

# Function to calculate a simple moving average (SMA) prediction
def moving_average_prediction(data, window=5):
    data['Predicted Price'] = data['Price'].rolling(window=window).mean()
    return data

# Function to train a linear regression model for price prediction
def linear_regression_prediction(data):
    data = data.dropna()
    X = np.array(range(len(data))).reshape(-1, 1)
    y = data['Price'].values
    model = LinearRegression()
    model.fit(X, y)
    data['Predicted Price'] = model.predict(X)
    return data, model

# Function to evaluate prediction performance
def evaluate_predictions(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)
    return mse, rmse, mape, r2

# Main function
def main(file_path, sheet_name):
    data = load_data(file_path, sheet_name)
    
    # Moving Average Prediction
    data = moving_average_prediction(data)
    print("Moving Average Prediction Applied\n")
    
    # Linear Regression Prediction
    data, model = linear_regression_prediction(data)
    print("Linear Regression Prediction Applied\n")
    
    # Evaluate Predictions
    data = data.dropna()
    mse, rmse, mape, r2 = evaluate_predictions(data['Price'], data['Predicted Price'])
    print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, R²: {r2:.2f}")
    
    # Plot actual vs predicted prices
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Price'], label='Actual Price', color='blue')
    plt.plot(data['Date'], data['Predicted Price'], label='Predicted Price', color='red', linestyle='dashed')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Stock Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

# Run the script if executed directly
if __name__ == "__main__":
    file_path = "LAB_04_D_NAVYA/lab_session_data.xlsx"
    sheet_name = "IRCTC Stock Price"
    main(file_path, sheet_name)