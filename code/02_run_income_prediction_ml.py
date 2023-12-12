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





#%% Get current working directory and parameters:

# Parameters
dataPath = 'J:/My Drive/PovertyPredictionRealTime/data'

freq = 'm'

date = '2023-12-11' #datetime.today().strftime('%Y-%m-%d')

#--------------

dpml = DataPreparationForML(freq=freq, dataPath=dataPath, date=date)

# Read dataset:

ml_dataset = dpml.read_consolidated_ml_dataset()


ml_dataset = ml_dataset.query('income_pc>0')

# First pass dropping all missing values:
ml_dataset_filtered = (ml_dataset.query('year >= 2016')
                                .query('year < 2020')
                                .sample(frac=1) # Random shuffle
                                .reset_index(drop=True) # Remove index
                                )

# ml_dataset_filtered_validation = ml_dataset_filtered.sample(int(ml_dataset_filtered.shape[0] * .075))

# ml_dataset_filtered = ml_dataset_filtered.loc[~ml_dataset_filtered.index.isin(ml_dataset_filtered_validation.index)]

# Y = np.log(ml_dataset_filtered.loc[:,dpml.depvar])
Y = ml_dataset_filtered.loc[:,'log_income_pc'].reset_index(drop=True)
# X = ml_dataset_filtered.loc[:,['log_income_pc_lagged', 'log_spend_pc_lagged'] + dpml.indepvars[2:]]
X = ml_dataset_filtered.loc[:,['log_income_pc_lagged', 'log_spend_pc_lagged'] + dpml.indepvars[2:] + dpml.indepvars_weather]
# X = ml_dataset_filtered.loc[:,['log_income_pc_lagged', 'log_spend_pc_lagged'] +  dpml.indepvars_weather]
X[dpml.indepvars_weather] = np.log(X[dpml.indepvars_weather] + 1)

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)


# group_means = ml_dataset_filtered.groupby(['year'])[dpml.indepvars].transform('mean')
# group_stds = ml_dataset_filtered.groupby(['year'])[dpml.indepvars].transform('std')

# Step 2: Standardize the variables
# X_standardized = (X - group_means) / group_stds
X_standardized = pd.DataFrame(X_imputed)
X_standardized.columns = X.columns
X_standardized['const'] = 1

# Split the model in validation data and train and testing data:
Y_validation = Y.iloc[:15000]
X_standardized_validation = X_standardized.iloc[:15000,:]

Y = Y.iloc[15000:]
X_standardized = X_standardized.iloc[15000:,:]


#%% Start machine learning models:

# define the models to test in a dictionary: including the hyperparameters to test

models = {
    "Linear Regression": (LinearRegression(), {}),
    "Lasso": (Lasso(), {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}),
    "Ridge": (Ridge(), {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}),
    # "Random Forest": (RandomForestRegressor(), {'n_estimators': [10, 50, 100, 200]}),
    # "Gradient Boosting": (GradientBoostingRegressor(), {'n_estimators': [10, 50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2, 0.5]})
}


n_folds = 5
# Dictionary to store grid search results
grid_search_results = {}

# Perform grid search with cross-validation for each model
for model_name, (model, params) in models.items():
    grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', return_train_score=True, n_jobs=4)
    grid_search.fit(X_standardized, Y)
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


# Get list of important variables according to Lasso:
X.columns[best_model.coef_[:-1] ==0]
X.columns[best_model.coef_[:-1] !=0]





# Use the best model to predict
predicted_income = best_model.predict(X_standardized_validation)

sns.histplot(predicted_income, color='red', kde=True, label='Predicted Income', stat='density')
sns.histplot(Y_validation, color='blue', kde=True, label='True Income', stat='density')
plt.xlim(4,12)
plt.legend()
# plt.savefig('../figures/prediction_vs_truth_log_income.pdf', bbox_inches='tight')
plt.show()
# Show the predictions

