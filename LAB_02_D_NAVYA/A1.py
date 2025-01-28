import pandas as pd
import numpy as np

file_path= "lab_session_data.xlsx"

# Function to load the data
def load_data(file_path):
    # A1. Please refer to the “Purchase Data” worksheet of Lab Session Data.xlsx. Please load the data
    return pd.read_excel(file_path)

# Function to extract matrices A and C
def extract_matrices(data):
    # and segregate them into 2 matrices A & C (following the nomenclature of AX = C).
    A = np.matrix(data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]])
    C = np.matrix(data[["Payment (Rs)"]])
    return A, C

# Function to get the dimensionality of the vector space
def get_dimensionality(A):
    # • What is the dimensionality of the vector space for this data?
    return A.shape[1]

# Function to get the number of vectors in the vector space
def get_num_vectors(A):
    # • How many vectors exist in this vector space?
    return A.shape[0]

# Function to calculate the rank of matrix A
def get_rank(A):
    # • What is the rank of Matrix A?
    return np.linalg.matrix_rank(A)

# Function to calculate the pseudo-inverse of A and use it to find the cost vector X
def calculate_costs(A, C):
    # • Using Pseudo-Inverse find the cost of each product available for sale.
    # (Suggestion: If you use Python, you can use numpy.linalg.pinv() function to get a
    # pseudo-inverse.)
    pseudo_inverse_A = np.linalg.pinv(A)
    model_vector_X = pseudo_inverse_A * C
    return model_vector_X

# Function to classify customers as RICH or POOR based on payment
def classify_customers(data):
    # Mark all customers (in “Purchase Data” table) with payments above Rs. 200 as RICH and others
    # as POOR. Develop a classifier model to categorize customers into RICH or POOR class based on
    # purchase behavior.
    classes = []
    for payment in data["Payment (Rs)"]:
        if payment > 200:
            classes.append("RICH")
        else:
            classes.append("POOR")
    data["Class"] = classes
    return data

# Main function to orchestrate the process
def main(file_path):
    # Main function to load data, process the matrices, and classify customers.
    # Load the data from the provided Excel file
    data = load_data(file_path)
    
    # Extract matrices A and C from the data
    A, C = extract_matrices(data)
    
    # Get the dimensionality of the vector space
    dimensionality = get_dimensionality(A)
    
    # Get the number of vectors in the vector space
    num_vectors = get_num_vectors(A)
    
    # Get the rank of matrix A
    rank_A = get_rank(A)
    
    # Calculate the cost of each product available for sale using the pseudo-inverse
    model_vector_X = calculate_costs(A, C)
    
    # Classify customers into RICH or POOR based on their payment
    classified_data = classify_customers(data)
    
    # Return results as a dictionary for easy access
    results = {
        "dimensionality": dimensionality,
        "num_vectors": num_vectors,
        "rank_A": rank_A,
        "model_vector_X": model_vector_X,
        "classified_data": classified_data
    }
    
    return results

# If this script is being executed directly, call the main function
if __name__ == "__main__":
    file_path = 'lab_session_data.xlsx'  # Path to the Excel file
    results = main(file_path)
    
    # Display the results
    print(f"Dimensionality of the vector space: {results['dimensionality']}")
    print(f"Number of vectors in the vector space: {results['num_vectors']}")
    print(f"Rank of Matrix A: {results['rank_A']}")
    print(f"Cost of each product available for sale using pseudo-inverse: \n{results['model_vector_X']}")
    print(f"Classified Customer Data: \n{results['classified_data'][['Customer', 'Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)', 'Class']]}")