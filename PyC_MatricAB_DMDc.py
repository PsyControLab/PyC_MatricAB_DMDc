import numpy as np
import pandas as pd

def PyC_MatricAB(df, id_col, X_cols, U_col, n):
    """
    Calculate A and B matrices for each entity in a complex system using Dynamic Mode Decomposition with Control (DMDc).

    :param df: DataFrame containing the data.
    :param id_col: Column name for unique entity identifiers.
    :param X_cols: List of column names representing state variables.
    :param U_col: Column name for the input variable.
    :param n: The dimension of the state variable (n in A_{n x n}).
    :return: DataFrame with entity_ids, A_matrice, and B_matrice.
    """
    # Get the unique entity IDs
    entity_ids = df[id_col].unique()

    # Dictionary to hold A and B matrices for each entity
    system_matrices = {}

    for entity_id in entity_ids:
        # Subset the DataFrame for the current entity
        x_id = df[df[id_col] == entity_id][X_cols]
        u_id = df[df[id_col] == entity_id][[U_col]]
        
        # Perform calculations
        x1 = np.log(x_id.iloc[:-1].T)
        x2 = np.log(x_id.iloc[1:].T)
        upsilon = u_id.iloc[:-1].T
        Omega = np.vstack((x1, upsilon))
        AB = np.dot(x2, np.linalg.pinv(Omega))
        A_matrix = AB[:, :-1]
        B_matrix = AB[:, -1].reshape(n, 1)

        # Store the A and B matrices
        system_matrices[entity_id] = {'A': A_matrix, 'B': B_matrix}

    # Lists to store A and B matrices for DataFrame
    entity_ids_list = []
    A_matrice_list = []
    B_matrice_list = []

    # Populate the lists
    for entity_id, matrices in system_matrices.items():
        entity_ids_list.append(entity_id)
        A_matrice_list.append(matrices['A'])
        B_matrice_list.append(matrices['B'])

    # Create a new DataFrame to hold the A and B matrices
    df_AB = pd.DataFrame({
        id_col: entity_ids_list,
        'A_matrice': A_matrice_list,
        'B_matrice': B_matrice_list,
    })

    return df_AB

# Example usage:
# Assuming your DataFrame is loaded into 'df'
# df = pd.read_csv('your_data.csv') # Load your data

# Use iloc to select columns 3 to 33 for state variables (Python indexing starts from 0)
#X_cols = df.iloc[:, 2:33].columns.tolist()

# Set the column name of your input variable
#U_col = 'Happy'  # Replace 'Happy' with the actual column name if different

# Call the function with the specified parameters
#df_AB = PyC_MatricAB(df, 'user_id', X_cols, U_col, 28)

# Display the first three rows of the resulting DataFrame
#df_AB.head(3)
