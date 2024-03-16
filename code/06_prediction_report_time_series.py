import os
import socket
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
from joblib import Parallel, delayed, dump, load
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.linear_model import Lasso
import numpy as np
from joblib import Parallel, delayed
from global_settings import global_settings


#%% Get current working directory and parameters:

# Parameters
freq = 'm'

date = '2024-03-15' #datetime.today().strftime('%Y-%m-%d')

settings = global_settings()

dataPath = settings.get_data_path()

#--------------

dpml = DataPreparationForML(freq=freq, dataPath=dataPath, date=date)

# 1. Read dataset:
#------------------

ml_dataset = (dpml.read_consolidated_ml_dataset()
                    .groupby(['ubigeo','conglome','vivienda','hogar_ine','year'])
                    .first()
                    .reset_index(drop=False)
                    )

ml_dataset['urbano'] = ml_dataset['strata'].isin([1,2,3,4,5]).astype(int)

ml_dataset['trend'] = ml_dataset['year'].astype(int) - 2011

ml_dataset['ubigeo_region'] = ml_dataset['ubigeo'].str[:4]

ml_dataset['ubigeo_provincia'] = ml_dataset['ubigeo'].str[:6]

ml_dataset['lima_metropolitana'] = ml_dataset['ubigeo_provincia'] == 'U-1501'

ml_dataset = dpml.input_missing_values(ml_dataset)


def add_random_shocks_by_region(ml_df, ml_df_train, error_col, region_col, shock_col, ubigeo_col):
    """
    Add a column of random shocks stratified by region to the DataFrame.
    Parameters:
    ml_df (DataFrame): The input DataFrame.
    income_col (str): The name of the column with predicted income values.
    region_col (str): The name of the column to store the region codes.
    shock_col (str): The name of the new column to store the random shocks.
    ubigeo_col (str): The name of the column containing ubigeo codes.
    Returns:
    DataFrame: The input DataFrame with the added column of random shocks.
    """
    # Copy the DataFrame to avoid modifying the original one
    df = ml_df.copy()
    # Extract region from ubigeo and create a new column for region
    df[region_col] = df[ubigeo_col].str[:4]
    # Initialize the random shock column with NaNs
    df[shock_col] = np.nan

    # Do the same for the train data so we can back out the std dev of predicted income in the region
    df_train = ml_df_train.query('year == 2016').copy()
    df_train[region_col] = df_train[ubigeo_col].str[:4]
    df_train[shock_col] = np.nan

    # Now, for each unique region, calculate the random shocks
    for region in df[region_col].unique():
        # Filter to get the predicted income for the region
        predicted_error_region = df.loc[df[region_col] == region, error_col]
        predicted_error_train_region_std = df_train.loc[df_train[region_col] == region, error_col].std()
        # Calculate the random shock for this region
        region_shock = np.random.normal(
            loc=0,
            scale=predicted_error_train_region_std ,  # scale based on the std dev of predicted income in the region
            size=predicted_error_region.shape[0]
        )
        # Assign the calculated shocks back to the main DataFrame
        df.loc[df[region_col] == region, shock_col] = region_shock
    return df


def group_variables_for_time_series(grouping_variables, df, frequency='yearly'):

    df = df.copy()

    household_weight = df['n_people']/df.groupby(grouping_variables)['n_people'].transform('sum')

    df['income_pc_weighted'] = df['income_pc'] * household_weight 
    df['income_pc_hat_weighted'] = df['income_pc_hat'] * household_weight 

    income_series = (df.groupby(grouping_variables)
                                .agg({
                                    'income_pc_weighted': 'sum', 
                                    'income_pc_hat_weighted': 'sum',
                                    'n_people': 'count'
                                    })
                                .reset_index()
                                )

    income_series['std_mean'] = income_series['income_pc_weighted']/np.sqrt(income_series['n_people'])
    income_series['std_hat_mean'] = income_series['income_pc_hat_weighted']/np.sqrt(income_series['n_people'])

    # Convert 'year' and 'month' to a datetime

    if frequency == 'yearly':
        income_series['date'] = pd.to_datetime(income_series[['year']].assign(MONTH=1,DAY=1))
    elif frequency == 'quarterly':
        income_series['date'] = pd.to_datetime(income_series.rename(columns={'quarter':'month'})[['year','month']].assign(DAY=1))

    return income_series


def group_porverty_rate_for_time_series(grouping_variables, df, frequency='yearly'):

    df = df.copy()

    household_weight = df['n_people']/df.groupby(grouping_variables)['n_people'].transform('sum')
    
    df['poor_685'] = (df['income_pc'] <= df['lp_685usd_ppp']) * household_weight
    df['poor_365'] = (df['income_pc'] <= df['lp_365usd_ppp']) * household_weight
    df['poor_215'] = (df['income_pc'] <= df['lp_215usd_ppp']) * household_weight
    df['poor_hat_685'] = (df['income_pc_hat'] <= df['lp_685usd_ppp']) * household_weight
    df['poor_hat_365'] = (df['income_pc_hat'] <= df['lp_365usd_ppp']) * household_weight
    df['poor_hat_215'] = (df['income_pc_hat'] <= df['lp_215usd_ppp']) * household_weight

    income_series = (df.groupby(grouping_variables)
                                .agg({
                                    'poor_685': 'sum', 
                                    'poor_365': 'sum',
                                    'poor_215': 'sum',
                                    'poor_hat_685': 'sum', 
                                    'poor_hat_365': 'sum',
                                    'poor_hat_215': 'sum',
                                    'n_people': 'count'
                                    })
                                .reset_index()
                                )
    income_series['std_685_mean'] = np.sqrt(income_series['poor_685']*(1-income_series['poor_685']))/np.sqrt(income_series['n_people'])
    income_series['std_365_mean'] = np.sqrt(income_series['poor_365']*(1-income_series['poor_365']))/np.sqrt(income_series['n_people'])
    income_series['std_215_mean'] = np.sqrt(income_series['poor_215']*(1-income_series['poor_215']))/np.sqrt(income_series['n_people'])
    # Convert 'year' and 'month' to a datetime

    if frequency == 'yearly':
        income_series['date'] = pd.to_datetime(income_series[['year']].assign(MONTH=1,DAY=1))
    elif frequency == 'quarterly':
        income_series['date'] = pd.to_datetime(income_series.rename(columns={'quarter':'month'})[['year','month']].assign(DAY=1))

    return income_series



#--------------------------------------------------------------------------
# 2. Obtain filtered dataset:
#--------------------------------------------------------------------------

year_end = 2021

ml_dataset_filtered_train = dpml.filter_ml_dataset(ml_dataset, year_end=year_end).query('year<=2016')

ml_dataset_filtered_validation = (
                                    dpml.filter_ml_dataset(ml_dataset, year_end=year_end)
                                        .query('year >= 2017')
                                        .query('year <= ' + str(year_end))
                                        .query('true_year==2016') # Keep only observations that correspond to 2016 data
                                    )

ml_dataset_filtered_true = (
                                    dpml.filter_ml_dataset(ml_dataset, year_end=year_end)
                                        .query('year >= 2017')
                                        .query('year <= ' + str(year_end))
                                        .query('true_year != 2016') # Keep observations that do not correspond to 2016 data
                                    )

Y_standardized_train, X_standardized_train, scaler_X_train, scaler_Y_train = dpml.get_depvar_and_features(ml_dataset_filtered_train)

Y_standardized_validation, X_standardized_validation, scaler_X_validation, scaler_Y_validation = dpml.get_depvar_and_features(ml_dataset_filtered_validation, scaler_X_train, scaler_Y_train)

#--------------------------------------------------------------------------
# 3. Load best model:
#--------------------------------------------------------------------------

best_model_lasso = dpml.load_ml_model(model_filename = 'best_weighted_lasso_model.joblib')

best_model_gb = dpml.load_ml_model(model_filename = 'best_weighted_gb_model.joblib')

#--------------------------------------------------------------------------
# 4. Keep variables used in Gradient Boosting model:
#--------------------------------------------------------------------------

# Train:
X_standardized_train =  X_standardized_train[X_standardized_train.columns[best_model_lasso.coef_ !=0]]
X_standardized_train['const'] = 1

# Validation:
X_standardized_validation =  X_standardized_validation[X_standardized_validation.columns[best_model_lasso.coef_ !=0]]
X_standardized_validation['const'] = 1

#--------------------------------------------------------------------------
# 5. Predict income
#--------------------------------------------------------------------------

# Train:
#-------
predicted_income = best_model_gb.predict(X_standardized_train)
ml_dataset_filtered_train['predicted_income'] = best_model_gb.predict(X_standardized_train) * scaler_Y_train.scale_[0] + scaler_Y_train.mean_[0]
ml_dataset_filtered_train['true_income'] = np.array(Y_standardized_train)* scaler_Y_train.scale_[0] + scaler_Y_train.mean_[0]
ml_dataset_filtered_train['predicted_error'] = ml_dataset_filtered_train['predicted_income'] - ml_dataset_filtered_train['true_income']

error_std = (ml_dataset_filtered_train['predicted_error']).std()

random_shock_train = np.array(add_random_shocks_by_region(
                                    ml_df=ml_dataset_filtered_train, 
                                    ml_df_train=ml_dataset_filtered_train,
                                    error_col='predicted_error', 
                                    region_col='region', 
                                    shock_col='random_shock', 
                                    ubigeo_col='ubigeo'
                                    ).random_shock
                                    )

ml_dataset_filtered_train['log_income_pc_hat'] = (predicted_income * scaler_Y_train.scale_[0] + scaler_Y_train.mean_[0]) + random_shock_train
ml_dataset_filtered_train['income_pc_hat'] = np.exp(ml_dataset_filtered_train['log_income_pc_hat']  ) 



# Validation:
#------------
predicted_income_validation = best_model_gb.predict(X_standardized_validation)
ml_dataset_filtered_validation['predicted_income'] = best_model_gb.predict(X_standardized_validation)* scaler_Y_train.scale_[0] + scaler_Y_train.mean_[0]
ml_dataset_filtered_validation['true_income'] = np.array(Y_standardized_validation)* scaler_Y_train.scale_[0] + scaler_Y_train.mean_[0]
ml_dataset_filtered_validation['predicted_error'] = ml_dataset_filtered_validation['predicted_income'] - ml_dataset_filtered_validation['true_income']

error_std = (ml_dataset_filtered_validation['predicted_error']).std()

# Add random shocks:
random_shock_validation = np.array(add_random_shocks_by_region(
                                    ml_df=ml_dataset_filtered_validation, 
                                    ml_df_train=ml_dataset_filtered_train,
                                    error_col='predicted_error', 
                                    region_col='region', 
                                    shock_col='random_shock', 
                                    ubigeo_col='ubigeo'
                                    ).random_shock
                                    )

ml_dataset_filtered_validation['log_income_pc_hat'] = (predicted_income_validation * scaler_Y_train.scale_[0] + scaler_Y_train.mean_[0]) + random_shock_validation
ml_dataset_filtered_validation['income_pc_hat'] = np.exp(ml_dataset_filtered_validation['log_income_pc_hat']  ) 


#--------------------------------------------------------------------------
# 5. Compiling both datasets and creating some variables:
#--------------------------------------------------------------------------

month_to_quarter = {1:1, 2:1, 3:1, 
                    4:4, 5:4, 6:4, 
                    7:7, 8:7, 9:7, 
                    10:10, 11:10, 12:10}

# Concatenate both datasets (train and validation):
df = pd.concat([ml_dataset_filtered_train, ml_dataset_filtered_validation], axis=0)

df['quarter'] = df['month'].map(month_to_quarter)

df['n_people'] = df['mieperho'] * df['pondera_i']


# Concatenate both datasets (train and true data):
df_true = pd.concat([ml_dataset_filtered_train, ml_dataset_filtered_true], axis=0)

df_true['quarter'] = df_true['month'].map(month_to_quarter)

df_true['n_people'] = df_true['mieperho'] * df_true['pondera_i']
