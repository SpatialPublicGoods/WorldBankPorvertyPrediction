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





#%% Get current working directory and parameters:

# Parameters
dataPath = 'J:/My Drive/PovertyPredictionRealTime/data'

freq = 'm'

date = '2023-12-12' #datetime.today().strftime('%Y-%m-%d')

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

ml_dataset_filtered['count_people'] = 1

conglome_count = ml_dataset_filtered.groupby(['conglome','year']).count().reset_index().loc[:,['conglome','year','count_people']]

conglome_count['count'] = conglome_count.groupby(['conglome']).transform('count')['year']

ml_dataset_filtered = ml_dataset_filtered.loc[ml_dataset_filtered['conglome'].isin(conglome_count.query('count==4').conglome.unique()),:]


# ml_dataset_filtered_validation = ml_dataset_filtered.sample(int(ml_dataset_filtered.shape[0] * .075))

# ml_dataset_filtered = ml_dataset_filtered.loc[~ml_dataset_filtered.index.isin(ml_dataset_filtered_validation.index)]

# Define dependent and independent variables:
Y = ml_dataset_filtered.loc[:,'log_income_pc'].reset_index(drop=True) # X = ml_dataset_filtered.loc[:,['log_income_pc_lagged'] + dpml.indepvars[2:]]
X = ml_dataset_filtered.loc[:,['log_income_pc_lagged'] + ['mieperho'] + dpml.indepvars[2:] + dpml.indepvars_weather]
X[dpml.indepvars_weather] = np.log(X[dpml.indepvars_weather] + 1)

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
variables_to_interact = ['mieperho'] + dpml.indepvars[2:] + dpml.indepvars_weather

# Create interaction terms
for var in variables_to_interact:
    for dummy in ubigeo_dummies.columns:
        interaction_term = X_standardized[var] * ubigeo_dummies[dummy]
        X_standardized[f"{var}_x_{dummy}"] = interaction_term


# Step 6: Split the model in validation data and train and testing data:
validation_sample_size = 5000
Y_standardized_validation = Y_standardized.iloc[:validation_sample_size]
X_standardized_validation = X_standardized.iloc[:validation_sample_size,:]

Y_standardized_train = Y_standardized.iloc[validation_sample_size:]
X_standardized_train = X_standardized.iloc[validation_sample_size:,:]


#%% Start machine learning models:

# define the models to test in a dictionary: including the hyperparameters to test

models = {
    "Linear Regression": (LinearRegression(), {}),
    "Lasso": (Lasso(), {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}),
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


# Get list of important variables according to Lasso:
X_standardized_train.columns[best_model.coef_[:-1] ==0]
X_standardized_train.columns[best_model.coef_[:-1] !=0]

X_standardized_train.columns[best_model.coef_ ==0]
X_standardized_train.columns[best_model.coef_ !=0].shape





# Use the best model to predict
predicted_income = best_model.predict(X_standardized)

sns.histplot(predicted_income, color='red', kde=True, label='Predicted Income', stat='density')
sns.histplot(Y_standardized_train, color='blue', kde=True, label='True Income', stat='density')
# plt.xlim(4,12)
plt.legend()
# plt.savefig('../figures/prediction_vs_truth_log_income.pdf', bbox_inches='tight')
plt.show()
# Show the predictions


# Use the best model to predict
predicted_income = best_model.predict(X_standardized_validation)

sns.histplot(predicted_income, color='red', kde=True, label='Predicted Income', stat='density')
sns.histplot(Y_standardized_validation*best_model.predict(X_standardized).std()/Y_standardized_validation.std(), color='blue', kde=True, label='True Income', stat='density')
# plt.xlim(4,12)
plt.legend()
# plt.savefig('../figures/prediction_vs_truth_log_income.pdf', bbox_inches='tight')
plt.show()
# Show the predictions




# Use the best model to predict
predicted_income_adjusted = predicted_income*Y_standardized_train.std()/best_model.predict(X_standardized).std()

sns.histplot(predicted_income_adjusted, color='red', kde=True, label='Predicted Income', stat='density')
sns.histplot(Y_standardized_validation, color='blue', kde=True, label='True Income', stat='density')
# plt.xlim(4,12)
plt.legend()
# plt.savefig('../figures/prediction_vs_truth_log_income.pdf', bbox_inches='tight')
plt.show()
# Show the predictions


# Use the best model to predict
sns.histplot(ml_dataset_filtered.loc[:validation_sample_size,'income_pc'], color='green', kde=True, label='Income', stat='density', bins=300) 
sns.histplot(np.exp(predicted_income_adjusted * Y.std() + Y.mean()) - 1, color='red', kde=True, label='Predicted Income', stat='density', bins=300)
plt.xlim(-1,2500)
plt.legend()
# plt.savefig('../figures/prediction_vs_truth_log_income.pdf', bbox_inches='tight')
plt.show()
# Show the predictions
