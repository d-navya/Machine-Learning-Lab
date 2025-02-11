import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

def load_data(file_path):
    df = pd.read_excel(file_path)
    drop_cols = ['Question', 'Correct_Code', 'Code_with_Error', 'code_processed',
                 'code_with_question', 'code_comment', 'code_with_solution', 'ast']
    df.drop(columns=drop_cols, inplace=True)
    df['Grade'] = (df['Final_Marks'] <= 5).astype(int)
    df.fillna(df.median(), inplace=True)
    
    X = df.drop(columns=['Grade'])
    Y = df['Grade']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

X_train, X_test, y_train, y_test = load_data("LAB_03_D_NAVYA/15 - C.xlsx")

def evaluate_knn(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)

    print(f"\nPerformance Metrics for k={k}:\n")
    
    print("\nTraining Data:")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))

    print("\nTest Data:")
    print(confusion_matrix(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))

    return knn

knn_model = evaluate_knn(k=3)