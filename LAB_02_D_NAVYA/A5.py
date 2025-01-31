import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Function to load data from the Excel file
def load_data(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

# Function to calculate mean and variance for price data
def calculate_mean_variance(data_column):
    population_mean = np.mean(data_column)
    population_variance = np.var(data_column)
    return population_mean, population_variance

# Function to identify data types and encode categorical variables
def identify_and_encode_data(data):
    # Identifying data types for each column
    print("Data types of each attribute:")
    print(data.dtypes)
    
    # Encoding categorical variables
    le = LabelEncoder()

    # Encoding nominal variables with One-Hot Encoding
    data = pd.get_dummies(data, drop_first=True)

    # Encoding ordinal variables with Label Encoding
    if 'Severity' in data.columns:
        data['Severity'] = le.fit_transform(data['Severity'])

    return data

# Function to check for missing values in the dataset
def check_missing_values(data):
    missing_values = data.isnull().sum()
    print(f"\nMissing values in each column:\n{missing_values}")
    missing_percentage = (missing_values / len(data)) * 100
    print(f"\nPercentage of missing values in each column:\n{missing_percentage}")

# Function to check for outliers using boxplot
def plot_boxplot(data_column, column_name):
    plt.figure(figsize=(8, 6))
    plt.boxplot(data_column)
    plt.title(f'Boxplot of {column_name}')
    plt.ylabel(column_name)
    plt.grid(True)
    plt.show()

# Function to calculate mean and variance for numeric columns
def calculate_numeric_summary(data):
    numeric_cols = data.select_dtypes(include=['float64', 'int64'])
    summary_stats = numeric_cols.describe().transpose()
    print(f"\nSummary statistics for numeric variables:\n{summary_stats}")
    return summary_stats

# Function to plot the distribution of numeric variables
def plot_histogram(data, column_name):
    plt.figure(figsize=(8, 6))
    plt.hist(data[column_name], bins=20, alpha=0.7)
    plt.title(f'Distribution of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Function to calculate outliers using IQR method
def detect_outliers_iqr(data_column):
    Q1 = data_column.quantile(0.25)
    Q3 = data_column.quantile(0.75)
    IQR = Q3 - Q1
    outliers = data_column[(data_column < (Q1 - 1.5 * IQR)) | (data_column > (Q3 + 1.5 * IQR))]
    print(f"\nOutliers detected using IQR method:\n{outliers}")
    return outliers

# Function to analyze and clean the dataset
def analyze_and_clean_data(file_path, sheet_name):
    # Load data
    data = load_data(file_path, sheet_name)

    # Identify and encode categorical variables
    data = identify_and_encode_data(data)

    # Calculate mean and variance of price data (assuming 'Price' column exists)
    if 'Price' in data.columns:
        population_mean, population_variance = calculate_mean_variance(data['Price'])
        print(f"\nPopulation mean price: {population_mean:.2f}")
        print(f"Population variance price: {population_variance:.2f}")
    
    # Check for missing values
    check_missing_values(data)

    # Check for outliers in numeric columns
    if 'Price' in data.columns:
        plot_boxplot(data['Price'], 'Price')
        detect_outliers_iqr(data['Price'])
    
    # Calculate mean and variance for numeric variables
    numeric_summary = calculate_numeric_summary(data)

    # Plot histogram for numeric variables (example: 'Price')
    if 'Price' in data.columns:
        plot_histogram(data, 'Price')
    
    return data

# Main function to orchestrate the entire analysis
def main(file_path, sheet_name):
    # Analyze and clean data
    cleaned_data = analyze_and_clean_data(file_path, sheet_name)
    print(f"\nData after cleaning and encoding:\n{cleaned_data.head()}")

# If this script is being executed directly, call the main function
if __name__ == "__main__":
    file_path = "/home/navya/Machine-Learning-Lab/LAB_02_D_NAVYA/lab_session_data.xlsx"  # Path to the Excel file
    sheet_name = "thyroid0387_UCI"  # Update this to the correct sheet name in your file
    main(file_path, sheet_name)


#A6
from sklearn.impute import SimpleImputer

def impute_data(data):
    # Identify numeric and categorical columns
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    # Impute numeric columns with mean or median
    for col in numeric_cols:
        # Check for outliers using IQR method
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[col][(data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))]

        # If the column contains outliers, use median for imputation
        if len(outliers) > 0:
            imputer = SimpleImputer(strategy='median')
        else:
            imputer = SimpleImputer(strategy='mean')
        
        data[col] = imputer.fit_transform(data[[col]])
    
    # Impute categorical columns with mode
    for col in categorical_cols:
        imputer = SimpleImputer(strategy='most_frequent')
        data[col] = imputer.fit_transform(data[[col]])

    return data

#A7
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalize_data(data, scaling_type="minmax"):
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    
    if scaling_type == "minmax":
        # Apply Min-Max Scaling
        min_max_scaler = MinMaxScaler()
        data[numeric_cols] = min_max_scaler.fit_transform(data[numeric_cols])
    elif scaling_type == "standard":
        # Apply Standard Scaling (Z-score)
        standard_scaler = StandardScaler()
        data[numeric_cols] = standard_scaler.fit_transform(data[numeric_cols])
    
    return data

#A8
def jaccard_and_smc(data):
    # Select the first two rows
    vector1 = data.iloc[0]
    vector2 = data.iloc[1]
    
    # Select only binary attributes (0 or 1 values)
    binary_cols = data.columns[data.isin([0, 1]).all()]
    
    vector1_binary = vector1[binary_cols]
    vector2_binary = vector2[binary_cols]
    
    # Calculate f11, f01, f10, f00
    f11 = sum((vector1_binary == 1) & (vector2_binary == 1))  # Both 1
    f01 = sum((vector1_binary == 0) & (vector2_binary == 1))  # 0 in vector1, 1 in vector2
    f10 = sum((vector1_binary == 1) & (vector2_binary == 0))  # 1 in vector1, 0 in vector2
    f00 = sum((vector1_binary == 0) & (vector2_binary == 0))  # Both 0
    
    # Jaccard Coefficient
    jc = f11 / (f01 + f10 + f11)
    
    # Simple Matching Coefficient
    smc = (f11 + f00) / (f00 + f01 + f10 + f11)
    
    return jc, smc
