import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 🔹 Load the dataset
file_path = "hi/15 - C.xlsx"  # Update this with your actual file path
df = pd.read_excel(file_path)

# 🔹 Extract numerical columns only
numerical_df = df.select_dtypes(include=['int64', 'float64'])

# 🔹 Extract Features & Labels
features = numerical_df.iloc[:, :-1].values  # Features: all but last numerical column
labels = numerical_df.iloc[:, -1].values  # Labels: last numerical column

# 🔹 Select Two Classes
class_1, class_2 = 10, 5 

# 🔹 Filter data for selected classes
class_1_vectors = features[labels == class_1]
class_2_vectors = features[labels == class_2]

# 🔹 Calculate Centroids
centroid_1 = np.mean(class_1_vectors, axis=0)
centroid_2 = np.mean(class_2_vectors, axis=0)

# 🔹 Calculate Spread (Standard Deviation)
spread_1 = np.std(class_1_vectors, axis=0)
spread_2 = np.std(class_2_vectors, axis=0)

# 🔹 Calculate Interclass Distance (Euclidean Distance between Centroids)
interclass_distance = np.linalg.norm(centroid_1 - centroid_2)

# 🔹 Print Results
print(f"\n🔹 Centroid for Class {class_1}:\n", centroid_1)
print(f"\n🔹 Spread (Standard Deviation) for Class {class_1}:\n", spread_1)
print(f"\n🔹 Centroid for Class {class_2}:\n", centroid_2)
print(f"\n🔹 Spread (Standard Deviation) for Class {class_2}:\n", spread_2)
print(f"\n🔹 Interclass Distance between Class {class_1} and Class {class_2}: {interclass_distance:.4f}")


# ─────────────────────────────────────────────────────────────
# 🔹 A2: Histogram, Mean, and Variance for a Selected Feature
# ─────────────────────────────────────────────────────────────

# 🔹 Select a Feature (Change index if needed)
feature_index = 0  # Modify if you want a different feature
feature_data = numerical_df.iloc[:, feature_index].dropna().values  # Drop NaN values

# 🔹 Calculate Mean & Variance
mean_value = np.mean(feature_data)
variance_value = np.var(feature_data)

# 🔹 Plot Histogram
plt.figure(figsize=(8, 5))
plt.hist(feature_data, bins=10, edgecolor='black', alpha=0.7)
plt.xlabel("Feature Values")
plt.ylabel("Frequency")
plt.title(f"Histogram for Feature: {numerical_df.columns[feature_index]}")
plt.show()

# 🔹 Print A2 Results
print(f"\n🔹 Feature Selected: {numerical_df.columns[feature_index]}")
print(f"\n🔹 Mean: {mean_value:.4f}")
print(f"\n🔹 Variance: {variance_value:.4f}")