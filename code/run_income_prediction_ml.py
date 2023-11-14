# Libraries
#--------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.impute import SimpleImputer


# Get current working directory and parameters:

dataPath = 'J:/My Drive/PovertyPredictionRealTime/data'

admin_data_path = os.path.join(dataPath, '1_raw/peru/big_data/admin')

working = '3_working'

clean = '4_clean'

freq = 'm'

date = '2023-11-13'


# Read dataset:

ml_dataset = pd.read_csv(os.path.join(dataPath, clean, 'ml_dataset_' + date +'.csv'), index_col=0)

ml_dataset.columns


# define dependent variable:
depvar = 'ingmo1hd'

# define independent variables:
indepvar_enaho = ['mieperho','income_pc_lagged','spend_pc_lagged']

indepvar_police_reports = ['Economic_Commercial_Offenses',
       'Family_Domestic_Issues', 'Fraud_Financial_Crimes',
       'Information_Cyber_Crimes', 'Intellectual_Property_Cultural_Heritage',
       'Miscellaneous_Offenses', 'Personal_Liberty_Violations',
       'Property_Real_Estate_Crimes', 'Public_Administration_Offenses',
       'Public_Order_Political_Crimes', 'Public_Safety_Health',
       'Sexual_Offenses', 'Theft_Robbery_Related_Crimes', 'Violence_Homicide']

indepvar_domestic_violence = ['cases_tot']

indepvar_cargo_vehicles = ['vehicles_tot', 'fab_5y_p', 'fab_10y_p', 'fab_20y_p',
       'fab_30y_p', 'pub_serv_p', 'payload_m', 'dry_weight_m',
       'gross_weight_m', 'length_m', 'width_m', 'height_m']

indepvars = indepvar_enaho + indepvar_police_reports + indepvar_domestic_violence + indepvar_cargo_vehicles



# First pass dropping all missing values:
ml_dataset_filtered = ml_dataset.query('year >= 2016').query('year < 2020').dropna()

Y = ml_dataset_filtered.loc[:,depvar]
X = ml_dataset_filtered.loc[:,indepvars]

group_means = ml_dataset_filtered.groupby(['ubigeo','year'])[indepvars].transform('mean')
group_stds = ml_dataset_filtered.groupby(['ubigeo','year'])[indepvars].transform('std')

# Step 2: Standardize the variables
X_standardized = (X - group_means) / group_stds






# imputer = SimpleImputer(strategy='mean')
# X_numerical = imputer.fit_transform(X)


#%% Start machine learning models:

# define the models to test in a dictionary: including the hyperparameters to test

models = {
    "Linear Regression": (LinearRegression(), {}),
    "Lasso": (Lasso(), {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}),
    "Ridge": (Ridge(), {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}),
    "Random Forest": (RandomForestRegressor(), {'n_estimators': [10, 50, 100, 200]}),
    "Gradient Boosting": (GradientBoostingRegressor(), {'n_estimators': [10, 50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2, 0.5]})
}


n_folds = 5
# Dictionary to store grid search results
grid_search_results = {}

# Perform grid search with cross-validation for each model
for model_name, (model, params) in models.items():
    grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', return_train_score=True, n_jobs=4)
    grid_search.fit(X, Y)
    grid_search_results[model_name] = grid_search

# Print the best parameters and corresponding RMSE for each model
for model_name, results in grid_search_results.items():
    best_rmse = np.sqrt(-results.best_score_)
    print(f"{model_name}: Best Params: {results.best_params_}, Best RMSE: {best_rmse:.3f}")

