import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "hi/15 - C.xlsx"  # Update this with your file path if needed
df = pd.read_excel(file_path)

# Convert all data to numeric (force conversion)
df = df.apply(pd.to_numeric, errors='coerce')

# Drop NaN values (optional, but useful to avoid issues)
df = df.dropna()

# Convert labels to integers (assuming the last column contains class labels)
features = df.iloc[:, :-1].values  # Extract features
labels = df.iloc[:, -1].astype(int).values  # Convert labels to integers

# --- A1: Intraclass Spread and Interclass Distance ---
def calculate_intraclass_interclass(features, labels):
    unique_classes = np.unique(labels)

    centroids = {}
    spreads = {}

    for cls in unique_classes:
        class_vectors = features[labels == cls]  # Ensure valid indexing
        centroids[cls] = np.mean(class_vectors, axis=0)
        spreads[cls] = np.std(class_vectors, axis=0)

    # Calculate interclass distance (Euclidean)
    if len(unique_classes) >= 2:
        centroid_values = list(centroids.values())
        interclass_distance = np.linalg.norm(centroid_values[0] - centroid_values[1])
        print(f"Interclass Distance: {interclass_distance}")

    return centroids, spreads

centroids, spreads = calculate_intraclass_interclass(features, labels)