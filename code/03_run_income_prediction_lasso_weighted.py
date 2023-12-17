import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from consolidate_ml_dataframe import DataPreparationForML
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed, dump, load
from sklearn.model_selection import KFold, GroupKFold, GridSearchCV, cross_val_score
from sklearn.base import clone
from sklearn.linear_model import Lasso
from ml_utils import CustomTimePanelSplit
import numpy as np
from joblib import Parallel, delayed




#%% Get current working directory and parameters:

# Parameters
dataPath = 'J:/My Drive/PovertyPredictionRealTime/data'

dataPath = '/home/fcalle0/datasets/WorldBankPovertyPrediction/'

freq = 'm'

date = '2023-12-15' #datetime.today().strftime('%Y-%m-%d')

#--------------

dpml = DataPreparationForML(freq=freq, dataPath=dataPath, date=date)

# Read dataset:

ml_dataset = (dpml.read_consolidated_ml_dataset()
                    .groupby(['ubigeo','conglome','vivienda','hogar_ine','year'])
                    .first()
                    .reset_index(drop=False)
                    )

month_to_quarter = {1:1, 2:1, 3:1, 
                    4:4, 5:4, 6:4, 
                    7:7, 8:7, 9:7, 
                    10:10, 11:10, 12:10}

ml_dataset['quarter'] = ml_dataset['month'].map(month_to_quarter)

ml_dataset['date'] = pd.to_datetime(ml_dataset[['year','quarter']].rename(columns={'quarter':'month'}).assign(DAY=1))

ml_dataset['urbano'] = ml_dataset['strata'].isin([1,2,3,4,5])

# Obtain filtered dataset:
ml_dataset_filtered_train = (dpml.filter_ml_dataset(ml_dataset)
                                .query('year<=2018')
                                .sort_values(['date','conglome'])
                                .reset_index(drop=True)
                                )


ml_dataset_filtered_train['cv_id'] = ml_dataset_filtered_train['ubigeo'].str[:4] + '-' + ml_dataset_filtered_train['urbano'].astype(int).astype(str) + '-' + ml_dataset_filtered_train['year'].astype(str)

Y_standardized_train, X_standardized_train, scaler_X_train, scaler_Y_train = dpml.get_depvar_and_features(ml_dataset_filtered_train)

#%% Run Lasso Regression (Regular Cross Validation):

# Define the model
lasso = Lasso()

# Define the parameter grid
param_grid = {'alpha': [0.0001, 0.0002, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 1]}
# param_grid = {'alpha': [0.0001, 0.001]}

# Define the number of folds for cross-validation
n_folds = 5

# Define the number of jobs for parallelization
n_jobs = 5  # Use -1 to use all processors

# Initialize variables to store the best model
best_score = float('inf')
best_params = None
best_model = None

# Define the cross-validation and model fitting procedure
def fit_model(train_index, test_index, model, params, X, y, weights):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    model_clone = clone(model).set_params(**params)
    model_clone.fit(X_train_fold, y_train_fold, sample_weight=weights[train_index])
    predictions = model_clone.predict(X_test_fold)
    rmse = np.sqrt(np.mean((predictions - y_test_fold) ** 2))
    return rmse, params, model_clone

# Custom cross-validation with sample weighting
# kf = KFold(n_splits=n_folds)

gkf = GroupKFold(n_splits=n_folds)


# Calculate weights for the entire dataset: higher for tail observations
std_dev = np.std(Y_standardized_train)
mean = np.mean(Y_standardized_train)
tails = (Y_standardized_train < mean - 2 * std_dev) | (Y_standardized_train > mean + 2 * std_dev)
weights = np.ones(Y_standardized_train.shape)
weights[tails] *= 5  # Increase the weights for the tail observations

# Perform grid search with parallel processing
results = Parallel(n_jobs=n_jobs)(
    delayed(fit_model)(train_index, test_index, lasso, {'alpha': alpha}, X_standardized_train, Y_standardized_train, weights)
    for alpha in param_grid['alpha'] for train_index, test_index in gkf.split(X_standardized_train, groups = ml_dataset_filtered_train['cv_id']) #kf.split(X_standardized_train)
)


# Extract the best parameters and model from the results
for rmse, params, model_clone in results:
    if rmse < best_score:
        best_score = rmse
        best_params = params
        best_model = model_clone

# Output the best results
print(f"Lasso: Best Params: {best_params}, Best RMSE: {best_score:.3f}")
if hasattr(best_model, 'coef_'):
    print(f"Coefficients of the best model: {best_model.coef_}")

# Save the Model
model_filename = 'best_weighted_lasso_model.joblib'
dump(best_model, 'best_weighted_lasso_model.joblib')
print(f"Model saved to {model_filename}")


#%% Get list of important variables according to Lasso:

# Get both the coefficients values and names 
lasso_coefs = best_model.coef_  # Replace with your actual coefficients
feature_names = X_standardized_train.columns  # Replace with your actual feature names

# Separate the base variables and interaction terms
base_vars = [f for f in feature_names if '_x_ubigeo_' not in f]
interaction_terms = [f for f in feature_names if '_x_ubigeo_' in f]

# Get unique categories from the interaction terms
categories = list(set(term.split('_x_ubigeo_')[1] for term in interaction_terms))
categories.sort(key=lambda x: int(x.split('-')[1]))  # Sort categories by the numerical part if necessary

# Create a DataFrame to store the coefficients
coef_matrix = pd.DataFrame(index=categories + ['Non-Interaction'], columns=base_vars)

# Assign the non-interaction term coefficients
for var in base_vars:
    coef_matrix.loc['Non-Interaction', var] = lasso_coefs[list(feature_names).index(var)]

# Assign the interaction term coefficients
for term in interaction_terms:
    var, category = term.split('_x_ubigeo_')
    coef_matrix.loc[category, var] = lasso_coefs[interaction_terms.index(term)]

# Replace NaN with 0 if there are any interaction terms missing for certain categories
coef_matrix.fillna(0, inplace=True)

# Plot the heatmap:
cmap = sns.light_palette("blue", as_cmap=True)
mask = coef_matrix.astype(float) == 0
plt.figure(figsize=(20, 10))  # Adjust figure size as needed
sns.heatmap(np.abs(coef_matrix.astype(float)), cmap=cmap, annot=False, mask=mask, 
            linewidths=0.1, linecolor='black')
# plt.ylabel('Categories (including Non-Interaction)')
# plt.xlabel('Variables')
plt.xticks(rotation=90,fontsize=8)
plt.yticks(rotation=0,fontsize=8)
plt.savefig('../figures/variable_contribution_lasso_regression_tails_weighted.pdf', bbox_inches='tight')
plt.clf()

#%%
# Use the best model to predict (LASSO REGRESSION)

ml_dataset_filtered_validation = (dpml.filter_ml_dataset(ml_dataset)
                        .query('year==2019')
                        .sort_values(['date','conglome'])
                        .reset_index(drop=True)
                        )

Y_standardized_validation, X_standardized_validation, scaler_X_validation, scaler_Y_validation = dpml.get_depvar_and_features(ml_dataset_filtered_validation,scaler_X_train, scaler_Y_train)

predicted_income_validation = best_model.predict(X_standardized_validation)

plt.clf()
plt.figure(figsize=(10, 10))
sns.histplot(pd.Series(predicted_income_validation).dropna(), 
             color='red', 
             kde=True, 
             fill=False, 
             element='step',
             label='Predicted Income', 
             stat='density')
sns.histplot(Y_standardized_validation.dropna(), 
             color='blue', 
             kde=True, 
             fill=False, 
             element='step',
             label='True Income', 
             stat='density')
# plt.xlim(0,2500)
plt.legend()
plt.savefig('../figures/fig0_prediction_vs_true_distribution_lasso_training_weighted.pdf', bbox_inches='tight')


print('End of code: 03_run_income_prediction_lasso_weighted.py')
