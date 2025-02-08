import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    """Removes unnecessary columns and converts the target variable into binary labels."""
    columns_to_drop = ['Question', 'Correct_Code', 'Code_with_Error', 'code_processed',
                       'code_with_question', 'code_comment', 'code_with_solution', 'ast']
    df.drop(columns=columns_to_drop, inplace=True)
    
    # Convert 'Final_Marks' into binary labels (0: low score, 1: high score)
    df['Grade'] = (df['Final_Marks'] > 5).astype(int)
    
    return df

def encode_error_types(df):
    """Extracts unique error types from 'Type_of_Error' and applies one-hot encoding."""
    unique_errors = set()
    
    for entry in df['Type_of_Error']:
        errors = entry.strip("[]").replace("'", "").split(", ")
        unique_errors.update(errors)
    
    for error in unique_errors:
        df[error] = df['Type_of_Error'].apply(lambda x: 1 if error in x else 0)
    
    df.drop(columns=['Type_of_Error'], inplace=True)
    return df

def load_and_process_data(file_path):
    """Loads the dataset, cleans it, and encodes categorical variables."""
    df = pd.read_excel(file_path)
    df = clean_data(df)
    df = encode_error_types(df)
    return df

def prepare_features_and_labels(df):
    """Handles missing values, scales features, and separates labels."""
    df.fillna(df.median(), inplace=True)
    
    features = df.drop(columns=['Grade'])
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    labels = df['Grade']
    return features, labels

def compute_class_statistics(features, labels, class_a, class_b):
    """Computes centroids, standard deviations, and interclass distance for two classes."""
    class_a_vectors = features[labels == class_a]
    class_b_vectors = features[labels == class_b]
    
    centroid_a = np.mean(class_a_vectors, axis=0)
    centroid_b = np.mean(class_b_vectors, axis=0)
    
    spread_a = np.std(class_a_vectors, axis=0)
    spread_b = np.std(class_b_vectors, axis=0)
    
    interclass_distance = np.linalg.norm(centroid_a - centroid_b)
    intra_class_distance = np.linalg.norm(spread_a - spread_b)
    
    return centroid_a, spread_a, centroid_b, spread_b, interclass_distance

def analyze_feature_distribution(df, feature_idx):
    """Computes and visualizes the distribution of a selected feature."""
    feature_values = df.iloc[:, feature_idx].dropna().values
    
    mean_val = np.mean(feature_values)
    variance_val = np.var(feature_values)
    
    plt.figure(figsize=(8, 5))
    plt.hist(feature_values, bins=10, edgecolor='black', alpha=0.7)
    plt.xlabel("Feature Values")
    plt.ylabel("Frequency")
    plt.title(f"Histogram for Feature: {df.columns[feature_idx]}")
    plt.show()
    
    return mean_val, variance_val

def compute_minkowski_distances(features, index_a, index_b, r_values):
    """Calculates Minkowski distance for different r values and plots the results."""
    vector_a = features[index_a]
    vector_b = features[index_b]
    
    distances = [minkowski(vector_a, vector_b, r) for r in r_values]
    
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
    file_path = "LAB_03_D_NAVYA/15 - C.xlsx"
    df = load_and_process_data(file_path)
    
    features, labels = prepare_features_and_labels(df)
    
    class_a, class_b = 0, 1
    centroid_a, spread_a, centroid_b, spread_b, interclass_dist = compute_class_statistics(features, labels, class_a, class_b)
    
    print(f"\nCentroid for Class {class_a}:\n", centroid_a)
    print(f"\nSpread (Standard Deviation) for Class {class_a}:\n", spread_a)
    print(f"\nCentroid for Class {class_b}:\n", centroid_b)
    print(f"\nSpread (Standard Deviation) for Class {class_b}:\n", spread_b)
    print(f"\nInterclass Distance between Class {class_a} and Class {class_b}: {interclass_dist:.4f}")
    
    feature_idx = 5
    mean_val, variance_val = analyze_feature_distribution(df, feature_idx)
    print(f"\nFeature: {df.columns[feature_idx]}")
    print(f"\nMean: {mean_val:.4f}")
    print(f"\nVariance: {variance_val:.4f}")
    
    index_a, index_b = 0, 1
    r_values = np.arange(1, 11)
    minkowski_dists = compute_minkowski_distances(features, index_a, index_b, r_values)
    
    for r, dist in zip(r_values, minkowski_dists):
        print(f"r = {r}: Minkowski Distance = {dist:.4f}")