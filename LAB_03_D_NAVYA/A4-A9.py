import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import minkowski
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

def load_dataset(file_path):
    """Loads dataset from an Excel file."""
    data = pd.read_excel(file_path)
    return pd.DataFrame(data)

def preprocess_data(df):
    """Prepares the dataset by dropping unnecessary columns and encoding the target variable."""
    drop_cols = ['Question', 'Correct_Code', 'Code_with_Error', 'code_processed',
                 'code_with_question', 'code_comment', 'code_with_solution', 'ast']
    df.drop(columns=drop_cols, inplace=True)
    df['Grade'] = (df['Final_Marks'] <= 5).astype(int)
    return df

def extract_errors(df):
    """Extracts unique error types from 'Type_of_Error' column and one-hot encodes them."""
    errors = set()
    for row in df['Type_of_Error']:
        error_list = row.strip("[]").replace("'", "").split(", ")
        errors.update(error_list)
    for error in errors:
        df[error] = df['Type_of_Error'].apply(lambda x: 1 if error in x else 0)
    df.drop(columns=['Type_of_Error'], inplace=True)
    return df, errors

def split_data(df):
    """Splits the dataset into training and testing sets after scaling the features."""
    df.fillna(df.median(), inplace=True)
    X = df.drop(columns=['Grade'])
    Y = df['Grade']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

def train_knn(X_train, y_train, k=3):
    """Trains a KNN classifier with the given k value."""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

def evaluate_model(knn, X_test, y_test):
    """Evaluates the KNN classifier on the test set."""
    accuracy = knn.score(X_test, y_test)
    predictions = knn.predict(X_test)
    return accuracy, predictions

def plot_accuracy(X_train, X_test, y_train, y_test):
    """Plots accuracy for different values of k (1 to 11)."""
    k_values = range(1, 12)
    accuracies = []
    for k in k_values:
        knn = train_knn(X_train, y_train, k)
        acc, _ = evaluate_model(knn, X_test, y_test)
        accuracies.append(acc)
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, accuracies, marker='o', linestyle='dashed', color='b')
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Accuracy")
    plt.title("KNN Accuracy vs. Number of Neighbors")
    plt.xticks(k_values)
    plt.grid()
    plt.show()

def print_confusion_matrix(y_test, y_pred):
    """Prints and visualizes the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    file_path = "LAB_03_D_NAVYA/15 - C.xlsx"
    df = load_dataset(file_path)
    df = preprocess_data(df)
    df, errors = extract_errors(df)
    print("Unique Errors:", errors)
    X_train, X_test, y_train, y_test = split_data(df)
    knn_model = train_knn(X_train, y_train, k=3)
    accuracy, predictions = evaluate_model(knn_model, X_test, y_test)
    print("KNN Accuracy (k=3):", accuracy)
    print("KNN Predictions (k=3):", predictions)
    test_vect = X_test[0].reshape(1, -1)
    predicted_class = knn_model.predict(test_vect)
    print(f"Predicted class for test vector: {predicted_class[0]}")
    knn_nn = train_knn(X_train, y_train, k=1)
    nn_accuracy, _ = evaluate_model(knn_nn, X_test, y_test)
    print("NN Accuracy (k=1):", nn_accuracy)
    plot_accuracy(X_train, X_test, y_train, y_test)
    print_confusion_matrix(y_test, predictions)
    if accuracy < 0.6:
        print("The model is likely underfitting (low accuracy). Consider using more features or a more complex model.")
    elif accuracy > 0.9 and nn_accuracy < 0.7:
        print("The model may be overfitting (high accuracy for k=3 but poor generalization). Consider using regularization.")
    else:
        print("The model is likely well-fitted (regular fit).")