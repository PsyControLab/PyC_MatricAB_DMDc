# PyC_MatricAB_DMDc
## Overview
Welcome to "PyC_MatricAB", an innovative Python repository created by the PsyControl Lab. Our focus is on analyzing complex systems within the realms of neuroscience and psychology. The centerpiece of this repository is the `PyC_MatricAB` function, which is meticulously designed to calculate A and B matrices in linear time-invariant systems. These calculations are instrumental in dissecting the dynamics of neural networks and psychological states, offering a window into the intricate workings of the human mind and behavior.

## Background
At PsyControl Lab, we are at the forefront of integrating diverse disciplines such as physics, engineering, neuroscience, and psychology. Our research harnesses the power of network control theory, machine learning, and AI to delve into the intricacies of psychiatric disorders and human behavior. The `PyC_MatricAB` function is a reflection of our commitment to advancing scientific understanding in these complex domains, bridging the gap between theoretical models and practical, data-driven applications.

## Installation
To get started with `PyC_MatricAB`, clone the repository and install the required Python packages:
```bash
git clone https://github.com/PsyControl/PyC_MatricAB_DMDc.git
cd PyC_MatricAB
pip install numpy pandas
```
## Usage
The PyC_MatricAB function is user-friendly and can be integrated easily into your data analysis workflow. Here's a step-by-step guide to using it:

```python
import pandas as pd
from PyC_MatricAB import PyC_MatricAB

# Example: Load your dataset into a pandas DataFrame
df = pd.read_csv('your_data.csv')

# Define parameters for the function
id_col = 'entity_id'  # This should be the name of the column with unique identifiers in your dataset
X_cols = ['state_var1', 'state_var2', 'state_var3']  # List of columns representing state variables
U_col = 'input_var'  # Column name for the input variable
n = 28  # The dimension of your state variable (n in A_{n x n})

# Execute the function to calculate the A and B matrices
df_AB = PyC_MatricAB(df, id_col, X_cols, U_col, n)

# Display the first few rows of the output
print(df_AB.head())
```
The function processes your data and returns a DataFrame with calculated A and B matrices for each unique entity, providing insights into the system's dynamics.
