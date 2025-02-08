import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """Prepares the dataset by dropping unnecessary columns and encoding the target variable."""
    drop_cols = ['Question', 'Correct_Code', 'Code_with_Error', 'code_processed',
                 'code_with_question', 'code_comment', 'code_with_solution', 'ast']
    df.drop(columns=drop_cols, inplace=True)

    # Convert 'Final_Marks' into a binary class label
    df['Grade'] = (df['Final_Marks'] <= 5).astype(int)

    return df


def extract_errors(df):
    """Extracts unique error types from 'Type_of_Error' column and one-hot encodes them."""
    errors = set()
    for row in df['Type_of_Error']:
        error_list = row.strip("[]").replace("'", "").split(", ")  # Cleaning up format
        errors.update(error_list)

    # One-hot encoding error types
    for error in errors:
        df[error] = df['Type_of_Error'].apply(lambda x: 1 if error in x else 0)

    df.drop(columns=['Type_of_Error'], inplace=True)
    
    return df

def load_data(file_path):
    """
    Load the dataset and preprocess it.
    """
    df = pd.read_excel(file_path)
    df = preprocess_data(df)
    df = extract_errors(df)
    return df

def extract_features_labels(df):
    """
    Extract features and labels from the numerical dataframe.
    """
    df.fillna(df.median(), inplace=True)  # Handling NaN values
    
    features = df.drop(columns=['Grade']) 
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    labels = df['Grade']
    return features, labels

def calculate_centroids_and_spread(features, labels, class_1, class_2):
    """
    Calculate centroids and spread (standard deviation) for two selected classes.
    """
    class_1_vectors = features[labels == class_1]
    class_2_vectors = features[labels == class_2]
    
    centroid_1 = np.mean(class_1_vectors, axis=0)
    centroid_2 = np.mean(class_2_vectors, axis=0)
    
    spread_1 = np.std(class_1_vectors, axis=0)
    spread_2 = np.std(class_2_vectors, axis=0)
    
    interclass_distance = np.linalg.norm(centroid_1 - centroid_2)
    intra_class_distance = np.linalg.norm(spread_1 - spread_2)
    
    return centroid_1, spread_1, centroid_2, spread_2, interclass_distance
def analyze_feature(numerical_df, feature_index):
    """
    Compute the mean and variance of a selected feature and plot its histogram.
    """
    feature_data = numerical_df.iloc[:, feature_index].dropna().values  # Drop NaN values
    
    mean_value = np.mean(feature_data)
    variance_value = np.var(feature_data)
    
    plt.figure(figsize=(8, 5))
    plt.hist(feature_data, bins=10, edgecolor='black', alpha=0.7)
    plt.xlabel("Feature Values")
    plt.ylabel("Frequency")
    plt.title(f"Histogram for Feature: {numerical_df.columns[feature_index]}")
    plt.show()
    
    return mean_value, variance_value

def compute_minkowski_distance(features, vec_1_index, vec_2_index, r_values):
    """
    Compute Minkowski distance between two feature vectors for different r values.
    """
    vec_1 = features[vec_1_index]
    vec_2 = features[vec_2_index]
    
    distances = [minkowski(vec_1, vec_2, r) for r in r_values]
    
    plt.figure(figsize=(8, 5))
    plt.plot(r_values, distances, marker='o', linestyle='-', color='b')
    plt.xlabel("r Value")
    plt.ylabel("Minkowski Distance")
    plt.title("Minkowski Distance vs r")
    plt.xticks(r_values)
    plt.grid()
    plt.show()
    
    return distances

if __name__ == "__main__":
    # Load dataset
    file_path = "LAB_03_D_NAVYA/15 - C.xlsx"  # Update this with the actual file path
    df = load_data(file_path)
    
    # Extract features and labels
    features, labels = extract_features_labels(df)
    
    # Define class labels for comparison
    # class_1 means the Final_Marks <= 5, class_2 means the Final_Marks > 5
    class_1, class_2 = 0, 1
    
    # Calculate centroids, spread, and interclass distance
    centroid_1, spread_1, centroid_2, spread_2, interclass_distance = calculate_centroids_and_spread(features, labels, class_1, class_2)
    
    # Display results
    print(f"\nCentroid for Class {class_1}:\n", centroid_1)
    print(f"\nSpread (Standard Deviation) for Class {class_1}:\n", spread_1)
    print(f"\nCentroid for Class {class_2}:\n", centroid_2)
    print(f"\nSpread (Standard Deviation) for Class {class_2}:\n", spread_2)
    print(f"\nInterclass Distance between Class {class_1} and Class {class_2}: {interclass_distance:.4f}")
    
    # Analyze a selected feature
    feature_index = 5  # Modify if needed
    mean_value, variance_value = analyze_feature(df, feature_index)
    
    print(f"\nFeature Selected: {df.columns[feature_index]}")
    print(f"\nMean: {mean_value:.4f}")
    print(f"\nVariance: {variance_value:.4f}")
    
    # Compute Minkowski distance
    vec_1_index, vec_2_index = 0, 1  # Modify indices if needed
    r_values = np.arange(1, 11)
    minkowski_distances = compute_minkowski_distance(features, vec_1_index, vec_2_index, r_values)
    
    # Display Minkowski distances
    for r, dist in zip(r_values, minkowski_distances):
        print(f"r = {r}: Minkowski Distance = {dist:.4f}")