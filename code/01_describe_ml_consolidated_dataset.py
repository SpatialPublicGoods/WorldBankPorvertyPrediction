import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from datetime import datetime
from consolidate_ml_dataframe import DataPreparationForML


dataPath = 'J:/My Drive/PovertyPredictionRealTime/data'

freq = 'm'

date = datetime.today().strftime('%Y-%m-%d')

#--------------

dpml = DataPreparationForML(freq=freq, dataPath=dataPath, date=date)

ml_dataset = dpml.read_consolidated_ml_dataset()

ml_dataset = ml_dataset.query('income_pc>0')

# First pass dropping all missing values:
ml_dataset_filtered = ml_dataset.query('year >= 2016').query('year < 2020')

Y = ml_dataset_filtered.loc[:,dpml.depvar]

X = ml_dataset_filtered.loc[:,[dpml.depvar] + dpml.indepvars]

numerical_descriptive_stats = X.describe()

print("Descriptive Statistics for Numerical Data:\n", numerical_descriptive_stats)


# Define the grid size
n_rows = 5
n_cols = 6

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))  # Adjust the figure size as needed
fig.tight_layout(pad=5.0)  # Adjust the spacing between subplots as needed

# Loop over the variables in X and create a histogram for each
for i, col in enumerate(X.columns):
    
    # Find the position of the subplot
    row_num = i // n_cols
    col_num = i % n_cols
    
    # Plot histogram for numerical data
    if X[col].dtype in ['int64', 'float64']:
        sns.histplot(X[col], kde=True, ax=axes[row_num, col_num])
        axes[row_num, col_num].set_title(f'')

# Show the entire grid plot
plt.savefig('../figures/histograms_ml_dataset_variables.pdf', bbox_inches='tight')
plt.show()



# Define the grid size for the subplot
n_rows = 5
n_cols = 6

# Create a figure with subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))  # Adjust the figure size as needed
fig.tight_layout(pad=5.0)  # Adjust the spacing between subplots as needed

# Loop over the variables in X and create a binscatter plot for each
for i, col in enumerate(X.columns):
    
    # Find the position of the subplot
    row_num = i // n_cols
    col_num = i % n_cols
    
    # Create binscatter plot for numerical data
    if X[col].dtype in ['int64', 'float64']:

        # Plot the binscatter plot
        sns.regplot(x=X[col], y=Y, ax=axes[row_num, col_num],
                    scatter_kws={'alpha':0.5}, line_kws={"color": "red"}, x_bins=10)
        
        axes[row_num, col_num].set_title(f'')

plt.show()