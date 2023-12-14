# Libraries
#--------------
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from consolidate_ml_dataframe import DataPreparationForML
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed




#%% Get current working directory and parameters:

# Parameters
dataPath = 'J:/My Drive/PovertyPredictionRealTime/data'

dataPath = '/home/fcalle0/datasets/WorldBankPovertyPrediction/'

freq = 'm'

date = '2023-12-12' #datetime.today().strftime('%Y-%m-%d')

#--------------

dpml = DataPreparationForML(freq=freq, dataPath=dataPath, date=date)

# Read dataset:

ml_dataset = dpml.read_consolidated_ml_dataset()

def filter_ml_dataset(ml_dataset):

    ml_dataset = ml_dataset.query('income_pc>0')

    # First pass dropping all missing values:
    ml_dataset_filtered = (ml_dataset.query('year >= 2016')
                                    .query('year < 2020')
                                    .sample(frac=1) # Random shuffle
                                    .reset_index(drop=True) # Remove index
                                    )

    ml_dataset_filtered['count_people'] = 1

    conglome_count = ml_dataset_filtered.groupby(['conglome','year']).count().reset_index().loc[:,['conglome','year','count_people']]

    conglome_count['count'] = conglome_count.groupby(['conglome']).transform('count')['year']

    ml_dataset_filtered = ml_dataset_filtered.loc[ml_dataset_filtered['conglome'].isin(conglome_count.query('count==4').conglome.unique()),:]

    ml_dataset_filtered = ml_dataset_filtered.dropna(subset='income_pc_lagged').reset_index(drop=True)

    return ml_dataset_filtered


ml_dataset_filtered = filter_ml_dataset(ml_dataset)

indepvar_column_names = ['mieperho'] + dpml.indepvars[2:] + dpml.indepvars_weather

# Define dependent and independent variables:
Y = ml_dataset_filtered.loc[:,'log_income_pc'].reset_index(drop=True) # X = ml_dataset_filtered.loc[:,['log_income_pc_lagged'] + dpml.indepvars[2:]]
X = ml_dataset_filtered.loc[:,['log_income_pc_lagged'] + ['mieperho'] + dpml.indepvars[2:] + dpml.indepvars_weather]
X[indepvar_column_names] = np.log(X[indepvar_column_names] + 1)

# Step 1: Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Step 2: Standardize X
scaler_X = StandardScaler()
X_standardized = scaler_X.fit_transform(X_imputed)
X_standardized = pd.DataFrame(X_standardized, columns=X.columns)
X_standardized['const'] = 1

# Step 3: Standardize Y
scaler_Y = StandardScaler()
Y_standardized = pd.Series(scaler_Y.fit_transform(Y.values.reshape(-1, 1)).flatten())  # Use flatten to convert it back to 1D array

# Step 4: Generate dummy variables for ubigeo and month: 
ubigeo_dummies = pd.get_dummies(ml_dataset_filtered['ubigeo'].str[:4], prefix='ubigeo', drop_first=True).reset_index(drop=True)
month_dummies = pd.get_dummies(ml_dataset_filtered['month'], prefix='month', drop_first=True).reset_index(drop=True)

# Step 5: Adding the dummy variables to X
X_standardized = pd.concat([X_standardized, ubigeo_dummies.astype(int), month_dummies.astype(int)], axis=1)

# Step 6: Create interaction terms:
variables_to_interact = ['log_income_pc_lagged'] + dpml.indepvars[2:] + dpml.indepvars_weather

# Create interaction terms
for var in variables_to_interact:
    for dummy in ubigeo_dummies.columns:
        interaction_term = X_standardized[var] * ubigeo_dummies[dummy]
        X_standardized[f"{var}_x_{dummy}"] = interaction_term


# Step 7: Split the model in validation data and train and testing data:
validation_sample_size = 5000
Y_standardized_validation = Y_standardized.iloc[:validation_sample_size]
X_standardized_validation = X_standardized.iloc[:validation_sample_size,:]

Y_standardized_train = Y_standardized.iloc[validation_sample_size:]
X_standardized_train = X_standardized.iloc[validation_sample_size:,:]


#%% Start machine learning models (No weighting of observations):

models = {
    # "Linear Regression": (LinearRegression(), {}),
    "Lasso": (Lasso(), {'alpha': [0.0001, 0.001, 0.01, 0.1, 10, 100]}),
    # "Ridge": (Ridge(), {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}),
    # "Random Forest": (RandomForestRegressor(), {'n_estimators': [10, 50, 100, 200]}),
    # "Gradient Boosting": (GradientBoostingRegressor(), {'n_estimators': [10, 50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2, 0.5]})
}


n_folds = 5
# Dictionary to store grid search results
grid_search_results = {}

# Perform grid search with cross-validation for each model
for model_name, (model, params) in models.items():
    grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', return_train_score=True, n_jobs=4)
    grid_search.fit(X_standardized_train, Y_standardized_train)
    grid_search_results[model_name] = grid_search

# Print the best parameters and corresponding RMSE for each model
for model_name, results in grid_search_results.items():
    best_rmse = np.sqrt(-results.best_score_)
    print(f"{model_name}: Best Params: {results.best_params_}, Best RMSE: {best_rmse:.3f}")



best_model_grid_search = grid_search_results['Lasso']
best_model = best_model_grid_search.best_estimator_
if hasattr(best_model, 'coef_'):
    print(f"Coefficients of the best model ({'Lasso'}): {best_model.coef_}")
elif hasattr(best_model, 'feature_importances_'):
    print(f"Feature importances of the best model ({'Lasso'}): {best_model.feature_importances_}")


#%% Start machine learning models (Weighting observations):




#%% Get list of important variables according to Lasso:

print(X_standardized_train.columns[best_model.coef_ !=0].shape)


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
plt.title('Lasso Coefficients Heatmap')
plt.ylabel('Categories (including Non-Interaction)')
plt.xlabel('Variables')
plt.xticks(rotation=90,fontsize=8)
plt.yticks(rotation=0,fontsize=8)
plt.savefig('../figures/variable_contribution_lasso_regression.pdf', bbox_inches='tight')
plt.show()
plt.clf()

#%%
# Use the best model to predict (LASSO REGRESSION)

predicted_income_train = best_model.predict(X_standardized_train)
sns.histplot(predicted_income_train, color='red', kde=True, label='Predicted Income', stat='density')
sns.histplot(Y_standardized_train, color='blue', kde=True, label='True Income', stat='density')
plt.legend()
plt.savefig('../figures/prediction_vs_true_distribution_lasso_training.pdf', bbox_inches='tight')
plt.show()

# Use the best model to predict
predicted_income_validation = best_model.predict(X_standardized_validation)

sns.histplot(predicted_income_validation, color='red', kde=True, label='Predicted Income', stat='density')
sns.histplot(Y_standardized_validation, color='blue', kde=True, label='True Income', stat='density')
plt.legend()
plt.savefig('../figures/prediction_vs_true_distribution_lasso_validation.pdf', bbox_inches='tight')
plt.show()


#%% Use features choosen by Lasso to predict income using Gradient Boosting:

XGB_standardized_train =  X_standardized_train[X_standardized_train.columns[best_model.coef_ !=0]]
XGB_standardized_validation =  X_standardized_validation[X_standardized_train.columns[best_model.coef_ !=0]]
XGB_standardized_train['const'] = 1
XGB_standardized_validation['const'] = 1


models = {
    # "Linear Regression": (LinearRegression(), {}),
    # "Lasso": (Lasso(), {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}),
    # "Ridge": (Ridge(), {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}),
    # "Random Forest": (RandomForestRegressor(), {'n_estimators': [10, 50, 100, 200]}),
    "Gradient Boosting": (GradientBoostingRegressor(), {'n_estimators': [150, 200, 250, 300, 350, 400], 'learning_rate': [0.1, 0.2, 0.5]})
}


n_folds = 5
# Dictionary to store grid search results
grid_search_results = {}

# Perform grid search with cross-validation for each model
for model_name, (model, params) in models.items():
    grid_search = GridSearchCV(model, 
                               params, 
                               cv=5, 
                               scoring='neg_mean_squared_error', 
                               return_train_score=True, 
                               n_jobs=8)
    grid_search.fit(XGB_standardized_train, Y_standardized_train)
    grid_search_results[model_name] = grid_search

# Print the best parameters and corresponding RMSE for each model
for model_name, results in grid_search_results.items():
    best_rmse = np.sqrt(-results.best_score_)
    print(f"{model_name}: Best Params: {results.best_params_}, Best RMSE: {best_rmse:.3f}")



best_model_grid_search = grid_search_results['Gradient Boosting']
best_model = best_model_grid_search.best_estimator_
if hasattr(best_model, 'coef_'):
    print(f"Coefficients of the best model ({'Lasso'}): {best_model.coef_}")
elif hasattr(best_model, 'feature_importances_'):
    print(f"Feature importances of the best model ({'Lasso'}): {best_model.feature_importances_}")



#%%
# Use the best model to predict (GRADIENT BOOSTING REGRESSION)

predicted_income_train = best_model.predict(X_standardized_train)

sns.histplot(predicted_income_train, color='red', kde=True, label='Predicted Income', stat='density')
sns.histplot(Y_standardized_train, color='blue', kde=True, label='True Income', stat='density')
plt.legend()
plt.show()

# Use the best model to predict
predicted_income_validation = best_model.predict(X_standardized_validation)

sns.histplot(predicted_income_validation, color='red', kde=True, label='Predicted Income', stat='density')
sns.histplot(Y_standardized_validation, color='blue', kde=True, label='True Income', stat='density')
plt.legend()
plt.savefig('../figures/prediction_vs_true_distribution_gradient_boosting.pdf', bbox_inches='tight')
plt.show()