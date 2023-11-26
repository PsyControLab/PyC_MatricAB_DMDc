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

# Example: Load your dataset into a pandas DataFrame
#df = pd.read_csv('your_data.csv')

# Define parameters for the function
#id_col = 'entity_id'  # This should be the name of the column with unique identifiers in your dataset
#X_cols = ['state_var1', 'state_var2', 'state_var3']  # List of columns representing state variables
#U_col = 'input'  # Column name for the input variable
#n = 28  # The dimension of your state variable (n in A_{n x n})

# Execute the function to calculate the A and B matrices
#df_AB = PyC_MatricAB(df, id_col, X_cols, U_col, n)

# Display the first few rows of the output
#print(df_AB.head())

# Merge the DataFrames based on the "user_id" column
# Grouping df by 'Group' and 'entity_ids' and taking the first occurrence
#df_grouped = df.groupby(['Group', 'entity_ids'], as_index=False).first()

# Merging df_grouped with df_AB on 'entity_ids'
#df_final = df_AB.merge(df_grouped, on='entity_ids')

# Printing the shape of the merged DataFrame
#print(df_final.shape)

# Displaying the first 3 rows of the merged DataFrame
#df_final.head(3)
