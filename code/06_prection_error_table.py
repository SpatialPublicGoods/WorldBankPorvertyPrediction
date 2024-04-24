import os
import socket
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
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

# Import custom classes
from global_settings import global_settings
from consolidate_ml_dataframe import DataPreparationForML
from post_estimation_ml_routines import PostEstimationRoutines
from generate_figures_for_report import GenerateFiguresReport

#%% Get current working directory and parameters:

# Parameters
freq = 'm'

date = '2024-04-22' #datetime.today().strftime('%Y-%m-%d')

settings = global_settings()

dataPath = settings.get_data_path()

# In your PC's cmd
# pip install sklearn=1.2.1

#--------------

dpml = DataPreparationForML(freq=freq, dataPath=dataPath, date=date)

postEstimation = PostEstimationRoutines()

figuresReport = GenerateFiguresReport()


#%%
# 1. Read dataset:
#------------------

ml_dataset = (dpml.read_consolidated_ml_dataset()
                    .groupby(['ubigeo','conglome','vivienda','hogar_ine','year'])
                    .first()
                    .reset_index(drop=False)
                    )

ml_dataset = postEstimation.generate_categorical_variables_for_analysis(ml_dataset)

#--------------------------------------------------------------------------
# 2. Obtain filtered dataset:
#--------------------------------------------------------------------------

base_year = dpml.base_year # Last year used for training

year_end = dpml.year_end


ml_dataset_filtered_train = dpml.filter_ml_dataset(ml_dataset, year_end=year_end).query('year<= ' + str(base_year))

# Validation dataset:
ml_dataset_filtered_validation = (
                                    dpml.filter_ml_dataset(ml_dataset, year_end=year_end)
                                        .query('year > '+ str(base_year))
                                        .query('year <= ' + str(year_end))
                                        .query('true_year=='+ str(base_year)) # Keep only observations that correspond to 2016 data
                                    )
# Validation dataset (World Bank version):
ml_dataset_filtered_validation_world_bank = (
                                    dpml.filter_ml_dataset(ml_dataset, year_end=year_end)
                                        .query('year > '+ str(base_year))
                                        .query('year <= ' + str(year_end))
                                        .query('true_year=='+ str(base_year)) # Keep only observations that correspond to 2016 data
                                    )
# True dataset:
ml_dataset_filtered_true = (
                                    dpml.filter_ml_dataset(ml_dataset, year_end=year_end)
                                        .query('year > '+ str(base_year))
                                        .query('year <= ' + str(year_end))
                                        .query('true_year !='+ str(base_year)) # Keep only observations that correspond to 2016 data
                                    )

Y_standardized_train, X_standardized_train, scaler_X_train, scaler_Y_train = dpml.get_depvar_and_features(ml_dataset_filtered_train)

Y_standardized_validation, X_standardized_validation, scaler_X_validation, scaler_Y_validation = dpml.get_depvar_and_features(ml_dataset_filtered_validation, scaler_X_train, scaler_Y_train)

#--------------------------------------------------------------------------
# 3. Load best model:
#--------------------------------------------------------------------------

best_model_lasso = postEstimation.load_ml_model(model_filename = 'best_weighted_lasso_model.joblib')

best_model_gb = postEstimation.load_ml_model(model_filename = 'best_weighted_gb_model.joblib')

#--------------------------------------------------------------------------
# 4. Keep variables used in Gradient Boosting model:
#--------------------------------------------------------------------------


# Train:
X_standardized_train =  postEstimation.get_variables_for_gb_model(best_model_lasso, X_standardized_train)

# Validation:
X_standardized_validation =  postEstimation.get_variables_for_gb_model(best_model_lasso, X_standardized_validation)

#--------------------------------------------------------------------------
# 5. Predict income
#--------------------------------------------------------------------------

# Get Y hat:
ml_dataset_filtered_train = postEstimation.add_predicted_income_to_dataframe(ml_dataset_filtered_train, X_standardized_train, Y_standardized_train, scaler_Y_train, best_model_gb)

ml_dataset_filtered_validation = postEstimation.add_predicted_income_to_dataframe(ml_dataset_filtered_validation, X_standardized_validation, Y_standardized_validation, scaler_Y_train, best_model_gb)

# Add shocks and compute income:
ml_dataset_filtered_train = postEstimation.add_shocks_and_compute_income(ml_dataset_filtered_train, 
                                                            ml_dataset_filtered_train)

ml_dataset_filtered_validation = postEstimation.add_shocks_and_compute_income(ml_dataset_filtered_validation, 
                                                            ml_dataset_filtered_train, 
                                                            )
# Compute predicted income (WB version):
ml_dataset_filtered_validation_world_bank = postEstimation.compute_predicted_income_world_bank(ml_dataset_filtered_validation_world_bank)

#--------------------------------------------------------------------------
# 5. Compiling both datasets and creating some variables:
#--------------------------------------------------------------------------


# Concatenate both datasets (train and validation):
df = pd.concat([ml_dataset_filtered_train, ml_dataset_filtered_validation], axis=0)

df['quarter'] = df['month'].map(dpml.month_to_quarter)

df['n_people'] = df['mieperho'] * df['pondera_i']


# Concatenate both datasets (train and true data):
df_true = pd.concat([ml_dataset_filtered_train, ml_dataset_filtered_true], axis=0)

df_true['quarter'] = df_true['month'].map(dpml.month_to_quarter)

df_true['n_people'] = df_true['mieperho'] * df_true['pondera_i']

# Concatenate both datasets (train and true data) World Bank Version:
df_wb = pd.concat([ml_dataset_filtered_train, ml_dataset_filtered_validation_world_bank], axis=0)

df_wb['quarter'] = df_wb['month'].map(dpml.month_to_quarter)

df_wb['n_people'] = df_wb['mieperho'] * df_wb['pondera_i']


# Get training data using the true level of income:
df.loc[df['year'] <= base_year, 'income_pc_hat'] = df.loc[df['year'] <= base_year, 'income_pc'] # Change income_pc_hat to income_pc for years <= s

df_wb.loc[df_wb['year'] <= base_year, 'income_pc_hat'] = df_wb.loc[df_wb['year'] <= base_year, 'income_pc'] # Change income_pc_hat to income_pc for years <= 2016

#--------------------------------------------------------------------------
# 6. Rural and Urban data
#--------------------------------------------------------------------------

grouping_variables = ['year']

income_series_pred = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df, frequency='yearly')
income_series_true = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df_true, frequency='yearly')
income_series_wb = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df_wb, frequency='yearly')

grouping_variables = ['year', 'urbano']

income_series_pred = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df, frequency='yearly')
income_series_true = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df_true, frequency='yearly')
income_series_wb = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df_wb, frequency='yearly')

# Split between urbano and rural:
income_series_pred_urban = income_series_pred.query('urbano==1')
income_series_pred_rural = income_series_pred.query('urbano==0')

income_series_true_urban = income_series_true.query('urbano==1')
income_series_true_rural = income_series_true.query('urbano==0')

income_series_wb_urban = income_series_wb.query('urbano==1')
income_series_wb_rural = income_series_wb.query('urbano==0')

#################
### Urban data
#################

df1 = income_series_true_urban[['year', 'n_people', 'poor_685']]
df2 = income_series_pred_urban[['year', 'poor_hat_685']]
df3 = income_series_wb_urban[['year', 'poor_hat_685']]

df_cat1 = pd.merge(df1,     df2, on='year', how='inner')
df_cat1 = pd.merge(df_cat1, df3, on='year', how='inner')

df_cat1['gb_urban_error'] = round(np.abs(df_cat1['poor_685'] - df_cat1['poor_hat_685_x']), 4)
df_cat1['wb_urban_error'] = round(np.abs(df_cat1['poor_685'] - df_cat1['poor_hat_685_y']), 4)

df_cat1 = df_cat1.iloc[ : , [0, 1, -2, -1]]

#################
### Rural data
#################

df1 = income_series_true_rural[['year', 'n_people', 'poor_685']]
df2 = income_series_pred_rural[['year', 'poor_hat_685']]
df3 = income_series_wb_rural[['year', 'poor_hat_685']]

df_cat2 = pd.merge(df1,     df2, on='year', how='inner')
df_cat2 = pd.merge(df_cat2, df3, on='year', how='inner')

df_cat2['gb_rural_error'] = round(np.abs(df_cat2['poor_685'] - df_cat2['poor_hat_685_x']), 4)
df_cat2['wb_rural_error'] = round(np.abs(df_cat2['poor_685'] - df_cat2['poor_hat_685_y']), 4)

df_cat2 = df_cat2.iloc[ : , [0, 1, -2, -1]]

#################
### Merging
#################

df_final = pd.merge(df_cat1, df_cat2, on='year', how='inner')

df_final['t_people'] = df_final['n_people_x'] + df_final['n_people_y']
df_final['pop_share_urban'] = round(df_final['n_people_x'] / df_final['t_people'], 4)
df_final['pop_share_rural'] = round(df_final['n_people_y'] / df_final['t_people'], 4)

df_final = df_final.iloc[ : , [0, 2, 3, 5, 6, -2, -1]]

df_final = df_final[df_final['year'] >= base_year + 1].reset_index(drop=True)

#################
### Reshape
#################

def reshape_data(df, category):
    years = df['year'].unique()
    reshaped_data = {}
    for year in years:
        reshaped_data[f'gb_error_{year}'] = df.loc[df['year'] == year, f'gb_{category}_error'].values
        reshaped_data[f'wb_error_{year}'] = df.loc[df['year'] == year, f'wb_{category}_error'].values
        reshaped_data[f'pop_share_{year}'] = df.loc[df['year'] == year, f'pop_share_{category}'].values
    return pd.DataFrame(reshaped_data, index=[category])

urban_data = reshape_data(df_final, 'urban')
rural_data = reshape_data(df_final, 'rural')

df_main = pd.concat([urban_data, rural_data])

#--------------------------------------------------------------------------
# 7. Male and Female data
#--------------------------------------------------------------------------

grouping_variables = ['year', 'hombre']

income_series_pred = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df, frequency='yearly')
income_series_true = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df_true, frequency='yearly')
income_series_wb = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df_wb, frequency='yearly')

# Split between urbano and rural:
income_series_pred_male = income_series_pred.query("hombre== 'Male' ")
income_series_pred_female = income_series_pred.query("hombre== 'Female' ")

income_series_true_male = income_series_true.query("hombre== 'Male' ")
income_series_true_female = income_series_true.query("hombre== 'Female' ")

income_series_wb_male = income_series_wb.query("hombre== 'Male' ")
income_series_wb_female = income_series_wb.query("hombre== 'Female' ")

#################
### Male data
#################

df1 = income_series_true_male[['year', 'n_people', 'poor_685']]
df2 = income_series_pred_male[['year', 'poor_hat_685']]
df3 = income_series_wb_male[['year', 'poor_hat_685']]

df_cat1 = pd.merge(df1,     df2, on='year', how='inner')
df_cat1 = pd.merge(df_cat1, df3, on='year', how='inner')

df_cat1['gb_male_error'] = round(np.abs(df_cat1['poor_685'] - df_cat1['poor_hat_685_x']), 4)
df_cat1['wb_male_error'] = round(np.abs(df_cat1['poor_685'] - df_cat1['poor_hat_685_y']), 4)

df_cat1 = df_cat1.iloc[ : , [0, 1, -2, -1]]

#################
### Female data
#################

df1 = income_series_true_female[['year', 'n_people', 'poor_685']]
df2 = income_series_pred_female[['year', 'poor_hat_685']]
df3 = income_series_wb_female[['year', 'poor_hat_685']]

df_cat2 = pd.merge(df1,     df2, on='year', how='inner')
df_cat2 = pd.merge(df_cat2, df3, on='year', how='inner')

df_cat2['gb_female_error'] = round(np.abs(df_cat2['poor_685'] - df_cat2['poor_hat_685_x']), 4)
df_cat2['wb_female_error'] = round(np.abs(df_cat2['poor_685'] - df_cat2['poor_hat_685_y']), 4)

df_cat2 = df_cat2.iloc[ : , [0, 1, -2, -1]]

#################
### Merging
#################

df_final = pd.merge(df_cat1, df_cat2, on='year', how='inner')

df_final['t_people'] = df_final['n_people_x'] + df_final['n_people_y']
df_final['pop_share_male']   = round(df_final['n_people_x'] / df_final['t_people'], 4)
df_final['pop_share_female'] = round(df_final['n_people_y'] / df_final['t_people'], 4)

df_final = df_final.iloc[ : , [0, 2, 3, 5, 6, -2, -1]]

df_final = df_final[df_final['year'] >= base_year + 1].reset_index(drop=True)

#################
### Reshape
#################

def reshape_data(df, category):
    years = df['year'].unique()
    reshaped_data = {}
    for year in years:
        reshaped_data[f'gb_error_{year}'] = df.loc[df['year'] == year, f'gb_{category}_error'].values
        reshaped_data[f'wb_error_{year}'] = df.loc[df['year'] == year, f'wb_{category}_error'].values
        reshaped_data[f'pop_share_{year}'] = df.loc[df['year'] == year, f'pop_share_{category}'].values
    return pd.DataFrame(reshaped_data, index=[category])

male_data   = reshape_data(df_final, 'male')
female_data = reshape_data(df_final, 'female')

df_aux = pd.concat([male_data, female_data])

#################
### Append
#################
df_main = pd.concat([df_main, df_aux])

#--------------------------------------------------------------------------
# 8. Formal and Informal data
#--------------------------------------------------------------------------

grouping_variables = ['year', 'categ_lab']

income_series_pred = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df, frequency='yearly')
income_series_true = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df_true, frequency='yearly')
income_series_wb = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df_wb, frequency='yearly')

# Split between urbano and rural:
income_series_pred_informal = income_series_pred.query("categ_lab== 'Informal' ")
income_series_pred_formal = income_series_pred.query("categ_lab== 'Not Informal' ")

income_series_true_informal = income_series_true.query("categ_lab== 'Informal' ")
income_series_true_formal = income_series_true.query("categ_lab== 'Not Informal' ")

income_series_wb_informal = income_series_wb.query("categ_lab== 'Informal' ")
income_series_wb_formal = income_series_wb.query("categ_lab== 'Not Informal' ")

#################
### Formal data
#################

df1 = income_series_true_formal[['year', 'n_people', 'poor_685']]
df2 = income_series_pred_formal[['year', 'poor_hat_685']]
df3 = income_series_wb_formal[['year', 'poor_hat_685']]

df_cat1 = pd.merge(df1,     df2, on='year', how='inner')
df_cat1 = pd.merge(df_cat1, df3, on='year', how='inner')

df_cat1['gb_formal_error'] = round(np.abs(df_cat1['poor_685'] - df_cat1['poor_hat_685_x']), 4)
df_cat1['wb_formal_error'] = round(np.abs(df_cat1['poor_685'] - df_cat1['poor_hat_685_y']), 4)

df_cat1 = df_cat1.iloc[ : , [0, 1, -2, -1]]

#################
### Informal data
#################

df1 = income_series_true_informal[['year', 'n_people', 'poor_685']]
df2 = income_series_pred_informal[['year', 'poor_hat_685']]
df3 = income_series_wb_informal[['year', 'poor_hat_685']]

df_cat2 = pd.merge(df1,     df2, on='year', how='inner')
df_cat2 = pd.merge(df_cat2, df3, on='year', how='inner')

df_cat2['gb_informal_error'] = round(np.abs(df_cat2['poor_685'] - df_cat2['poor_hat_685_x']), 4)
df_cat2['wb_informal_error'] = round(np.abs(df_cat2['poor_685'] - df_cat2['poor_hat_685_y']), 4)

df_cat2 = df_cat2.iloc[ : , [0, 1, -2, -1]]

#################
### Merging
#################

df_final = pd.merge(df_cat1, df_cat2, on='year', how='inner')

df_final['t_people'] = df_final['n_people_x'] + df_final['n_people_y']
df_final['pop_share_formal']   = round(df_final['n_people_x'] / df_final['t_people'], 4)
df_final['pop_share_informal'] = round(df_final['n_people_y'] / df_final['t_people'], 4)

df_final = df_final.iloc[ : , [0, 2, 3, 5, 6, -2, -1]]

df_final = df_final[df_final['year'] >= base_year + 1].reset_index(drop=True)

#################
### Reshape
#################

def reshape_data(df, category):
    years = df['year'].unique()
    reshaped_data = {}
    for year in years:
        reshaped_data[f'gb_error_{year}'] = df.loc[df['year'] == year, f'gb_{category}_error'].values
        reshaped_data[f'wb_error_{year}'] = df.loc[df['year'] == year, f'wb_{category}_error'].values
        reshaped_data[f'pop_share_{year}'] = df.loc[df['year'] == year, f'pop_share_{category}'].values
    return pd.DataFrame(reshaped_data, index=[category])

formal_data   = reshape_data(df_final, 'formal')
informal_data = reshape_data(df_final, 'informal')

df_aux = pd.concat([formal_data, informal_data])

#################
### Append
#################
df_main = pd.concat([df_main, df_aux])

#--------------------------------------------------------------------------
# 9. Elementary and Superior Education data
#--------------------------------------------------------------------------

grouping_variables = ['year', 'educ']

income_series_pred = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df, frequency='yearly')
income_series_true = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df_true, frequency='yearly')
income_series_wb = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df_wb, frequency='yearly')

# Split between urbano and rural:
income_series_pred_elementary = income_series_pred.query("educ== 'Elementary' ")
income_series_pred_superior = income_series_pred.query("educ== 'Superior' ")

income_series_true_elementary = income_series_true.query("educ== 'Elementary' ")
income_series_true_superior = income_series_true.query("educ== 'Superior' ")

income_series_wb_elementary = income_series_wb.query("educ== 'Elementary' ")
income_series_wb_superior = income_series_wb.query("educ== 'Superior' ")

#################
### Elementary data
#################

df1 = income_series_true_elementary[['year', 'n_people', 'poor_685']]
df2 = income_series_pred_elementary[['year', 'poor_hat_685']]
df3 = income_series_wb_elementary[['year', 'poor_hat_685']]

df_cat1 = pd.merge(df1,     df2, on='year', how='inner')
df_cat1 = pd.merge(df_cat1, df3, on='year', how='inner')

df_cat1['gb_elementary_error'] = round(np.abs(df_cat1['poor_685'] - df_cat1['poor_hat_685_x']), 4)
df_cat1['wb_elementary_error'] = round(np.abs(df_cat1['poor_685'] - df_cat1['poor_hat_685_y']), 4)

df_cat1 = df_cat1.iloc[ : , [0, 1, -2, -1]]

#################
### Superior data
#################

df1 = income_series_true_superior[['year', 'n_people', 'poor_685']]
df2 = income_series_pred_superior[['year', 'poor_hat_685']]
df3 = income_series_wb_superior[['year', 'poor_hat_685']]

df_cat2 = pd.merge(df1,     df2, on='year', how='inner')
df_cat2 = pd.merge(df_cat2, df3, on='year', how='inner')

df_cat2['gb_superior_error'] = round(np.abs(df_cat2['poor_685'] - df_cat2['poor_hat_685_x']), 4)
df_cat2['wb_superior_error'] = round(np.abs(df_cat2['poor_685'] - df_cat2['poor_hat_685_y']), 4)

df_cat2 = df_cat2.iloc[ : , [0, 1, -2, -1]]

#################
### Merging
#################

df_final = pd.merge(df_cat1, df_cat2, on='year', how='inner')

df_final['t_people'] = df_final['n_people_x'] + df_final['n_people_y']
df_final['pop_share_elementary'] = round(df_final['n_people_x'] / df_final['t_people'], 4)
df_final['pop_share_superior']   = round(df_final['n_people_y'] / df_final['t_people'], 4)

df_final = df_final.iloc[ : , [0, 2, 3, 5, 6, -2, -1]]

df_final = df_final[df_final['year'] >= base_year + 1].reset_index(drop=True)

#################
### Reshape
#################

def reshape_data(df, category):
    years = df['year'].unique()
    reshaped_data = {}
    for year in years:
        reshaped_data[f'gb_error_{year}'] = df.loc[df['year'] == year, f'gb_{category}_error'].values
        reshaped_data[f'wb_error_{year}'] = df.loc[df['year'] == year, f'wb_{category}_error'].values
        reshaped_data[f'pop_share_{year}'] = df.loc[df['year'] == year, f'pop_share_{category}'].values
    return pd.DataFrame(reshaped_data, index=[category])

elementary_data = reshape_data(df_final, 'elementary')
superior_data   = reshape_data(df_final, 'superior')

df_aux = pd.concat([elementary_data, superior_data])

#################
### Append
#################
df_main = pd.concat([df_main, df_aux])

#--------------------------------------------------------------------------
# 10. Number of children data
#--------------------------------------------------------------------------

grouping_variables = ['year', 'n_children']

income_series_pred = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df, frequency='yearly')
income_series_true = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df_true, frequency='yearly')
income_series_wb = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df_wb, frequency='yearly')

# Split between urbano and rural:
income_series_true_nchild0 = income_series_true.query("n_children== '0' ")
income_series_true_nchild1 = income_series_true.query("n_children== '1' ")
income_series_true_nchild2 = income_series_true.query("n_children== '2' ")
income_series_true_nchild3 = income_series_true.query("n_children== '3 more' ")

income_series_pred_nchild0 = income_series_pred.query("n_children== '0' ")
income_series_pred_nchild1 = income_series_pred.query("n_children== '1' ")
income_series_pred_nchild2 = income_series_pred.query("n_children== '2' ")
income_series_pred_nchild3 = income_series_pred.query("n_children== '3 more' ")

income_series_wb_nchild0 = income_series_wb.query("n_children== '0' ")
income_series_wb_nchild1 = income_series_wb.query("n_children== '1' ")
income_series_wb_nchild2 = income_series_wb.query("n_children== '2' ")
income_series_wb_nchild3 = income_series_wb.query("n_children== '3 more' ")

#################
### 0 children data
#################

df1 = income_series_true_nchild0[['year', 'n_people', 'poor_685']]
df2 = income_series_pred_nchild0[['year', 'poor_hat_685']]
df3 = income_series_wb_nchild0[['year', 'poor_hat_685']]

df_cat1 = pd.merge(df1,     df2, on='year', how='inner')
df_cat1 = pd.merge(df_cat1, df3, on='year', how='inner')

df_cat1['gb_nchild0_error'] = round(np.abs(df_cat1['poor_685'] - df_cat1['poor_hat_685_x']), 4)
df_cat1['wb_nchild0_error'] = round(np.abs(df_cat1['poor_685'] - df_cat1['poor_hat_685_y']), 4)

df_cat1 = df_cat1.iloc[ : , [0, 1, -2, -1]]

#################
### 1 children data
#################

df1 = income_series_true_nchild1[['year', 'n_people', 'poor_685']]
df2 = income_series_pred_nchild1[['year', 'poor_hat_685']]
df3 = income_series_wb_nchild1[['year', 'poor_hat_685']]

df_cat2 = pd.merge(df1,     df2, on='year', how='inner')
df_cat2 = pd.merge(df_cat2, df3, on='year', how='inner')

df_cat2['gb_nchild1_error'] = round(np.abs(df_cat2['poor_685'] - df_cat2['poor_hat_685_x']), 4)
df_cat2['wb_nchild1_error'] = round(np.abs(df_cat2['poor_685'] - df_cat2['poor_hat_685_y']), 4)

df_cat2 = df_cat2.iloc[ : , [0, 1, -2, -1]]

#################
### 2 children data
#################

df1 = income_series_true_nchild2[['year', 'n_people', 'poor_685']]
df2 = income_series_pred_nchild2[['year', 'poor_hat_685']]
df3 = income_series_wb_nchild2[['year', 'poor_hat_685']]

df_cat3 = pd.merge(df1,     df2, on='year', how='inner')
df_cat3 = pd.merge(df_cat3, df3, on='year', how='inner')

df_cat3['gb_nchild2_error'] = round(np.abs(df_cat3['poor_685'] - df_cat3['poor_hat_685_x']), 4)
df_cat3['wb_nchild2_error'] = round(np.abs(df_cat3['poor_685'] - df_cat3['poor_hat_685_y']), 4)

df_cat3 = df_cat3.iloc[ : , [0, 1, -2, -1]]

#################
### 3 children data
#################

df1 = income_series_true_nchild3[['year', 'n_people', 'poor_685']]
df2 = income_series_pred_nchild3[['year', 'poor_hat_685']]
df3 = income_series_wb_nchild3[['year', 'poor_hat_685']]

df_cat4 = pd.merge(df1,     df2, on='year', how='inner')
df_cat4 = pd.merge(df_cat4, df3, on='year', how='inner')

df_cat4['gb_nchild3_error'] = round(np.abs(df_cat4['poor_685'] - df_cat4['poor_hat_685_x']), 4)
df_cat4['wb_nchild3_error'] = round(np.abs(df_cat4['poor_685'] - df_cat4['poor_hat_685_y']), 4)

df_cat4 = df_cat4.iloc[ : , [0, 1, -2, -1]]

#################
### Merging
#################

df_final = pd.merge(df_cat1,  df_cat2, on='year', how='inner', suffixes=['0', '1'])
df_final = pd.merge(df_final, df_cat3, on='year', how='inner')
df_final = pd.merge(df_final, df_cat4, on='year', how='inner', suffixes=['2', '3'])

df_final['t_people'] = df_final['n_people0'] + df_final['n_people1'] + df_final['n_people2'] + df_final['n_people3']
df_final['pop_share_nchild0'] = round(df_final['n_people0'] / df_final['t_people'], 4)
df_final['pop_share_nchild1'] = round(df_final['n_people1'] / df_final['t_people'], 4)
df_final['pop_share_nchild2'] = round(df_final['n_people2'] / df_final['t_people'], 4)
df_final['pop_share_nchild3'] = round(df_final['n_people3'] / df_final['t_people'], 4)

df_final = df_final.iloc[ : , [0, 2, 3, 5, 6, 8, 9, 11, 12, -4, -3, -2, -1]]

df_final = df_final[df_final['year'] >= base_year + 1].reset_index(drop=True)

#################
### Reshape
#################

def reshape_data(df, category):
    years = df['year'].unique()
    reshaped_data = {}
    for year in years:
        reshaped_data[f'gb_error_{year}'] = df.loc[df['year'] == year, f'gb_{category}_error'].values
        reshaped_data[f'wb_error_{year}'] = df.loc[df['year'] == year, f'wb_{category}_error'].values
        reshaped_data[f'pop_share_{year}'] = df.loc[df['year'] == year, f'pop_share_{category}'].values
    return pd.DataFrame(reshaped_data, index=[category])

nchild0_data = reshape_data(df_final, 'nchild0')
nchild1_data = reshape_data(df_final, 'nchild1')
nchild2_data = reshape_data(df_final, 'nchild2')
nchild3_data = reshape_data(df_final, 'nchild3')

df_aux = pd.concat([nchild0_data, nchild1_data, nchild2_data, nchild3_data])

#################
### Append
#################
df_main = pd.concat([df_main, df_aux])

#--------------------------------------------------------------------------
# 11. Export table
#--------------------------------------------------------------------------

path_github = "C:/Users/franc/OneDrive/Documents/GitHub/Chicagobooth/WorldBankPorvertyPrediction/tables"
file_name   = "prediction_error_table.csv"
df_main.to_csv(os.path.join(path_github, file_name), index=True)











