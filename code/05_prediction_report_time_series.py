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


#%% Figure 5 (fig5_average_income_time_series): 
# Time series of average income (Yearly)
#----------------------------------------------------


grouping_variables = ['year']

income_series_pred  = postEstimation.group_variables_for_time_series(grouping_variables = grouping_variables, df=df, frequency='yearly')
income_series_true  = postEstimation.group_variables_for_time_series(grouping_variables = grouping_variables, df=df_true, frequency='yearly')
income_series_wb    = postEstimation.group_variables_for_time_series(grouping_variables = grouping_variables, df=df_wb, frequency='yearly')

# Plotting:
plt.clf()
plt.figure(figsize=(10, 10))

# Plotting the means with standard deviation
plt.errorbar(income_series_true['date'], income_series_true['income_pc_weighted'], yerr=income_series_true['std_mean'], 
            label='True Income', color=settings.color1, fmt='-')
plt.errorbar(income_series_pred['date'], income_series_pred['income_pc_hat_weighted'], yerr=income_series_pred['std_hat_mean'], 
            label='Predicted Income (GB)', color=settings.color2, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb['date'], income_series_wb['income_pc_hat_weighted'], yerr=income_series_wb['std_hat_mean'], 
            label='Predicted Income (WB)', color=settings.color3, fmt='-.', linestyle='-.')  # Adjust linestyle if needed

plt.xlabel('Date')
plt.ylabel('Income')
plt.legend()
plt.savefig('../figures/fig5_average_income_time_series.pdf', bbox_inches='tight')

print('Figure 5 saved')




#%% Figure 6 (fig6_average_income_time_series_by_area): 
# Time series average income plot by area (Yearly)
#----------------------------------------------------

grouping_variables = ['year','urbano']

income_series_pred = postEstimation.group_variables_for_time_series(grouping_variables = grouping_variables, df=df, frequency='yearly')
income_series_true = postEstimation.group_variables_for_time_series(grouping_variables = grouping_variables, df=df_true, frequency='yearly')
income_series_wb   = postEstimation.group_variables_for_time_series(grouping_variables = grouping_variables, df=df_wb, frequency='yearly')

# Income series for urban areas both true and predicted
income_series_true_urban = income_series_true.query('urbano==1')
income_series_pred_urban = income_series_pred.query('urbano==1')
income_series_wb_urban = income_series_wb.query('urbano==1')

# Income series for rural areas both true and predicted
income_series_true_rural = income_series_true.query('urbano==0')
income_series_pred_rural = income_series_pred.query('urbano==0')
income_series_wb_rural = income_series_wb.query('urbano==0')

# Plotting the means with standard deviation
# Urbano
plt.clf()
plt.figure(figsize=(10, 10))
plt.errorbar(income_series_true_urban['date'], income_series_true_urban['income_pc_weighted'], yerr=income_series_true_urban['std_mean'], 
            label='True Urban', color=settings.color1, fmt='-')
plt.errorbar(income_series_pred_urban['date'], income_series_pred_urban['income_pc_hat_weighted'], yerr=income_series_pred_urban['std_hat_mean'], 
            label='Predicted Urban (GB)', color=settings.color1, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_urban['date'], income_series_wb_urban['income_pc_hat_weighted'], yerr=income_series_wb_urban['std_hat_mean'], 
            label='Predicted Urban (WB)', color=settings.color1, fmt='+', linestyle=':')  # Adjust linestyle if needed
# Rural
plt.errorbar(income_series_true_rural['date'], income_series_true_rural['income_pc_weighted'], yerr=income_series_true_rural['std_mean'], 
            label='True Rural', color=settings.color4, fmt='-')
plt.errorbar(income_series_pred_rural['date'], income_series_pred_rural['income_pc_hat_weighted'], yerr=income_series_pred_rural['std_hat_mean'], 
            label='Predicted Rural (GB)', color=settings.color4, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_rural['date'], income_series_wb_rural['income_pc_hat_weighted'], yerr=income_series_wb_rural['std_hat_mean'], 
            label='Predicted Rural (WB)', color=settings.color4, fmt='--', linestyle=':')  # Adjust linestyle if needed

plt.xlabel('Date')
plt.ylabel('Income')
plt.legend(loc='lower left')
plt.savefig('../figures/fig6_average_income_time_series_by_area.pdf', bbox_inches='tight')

print('Figure 6 saved')

#%% Figure 7: (fig7_average_income_time_series_quarterly)
# Time series average income (Quarterly)
#----------------------------------------------------

grouping_variables = ['year','quarter']

income_series_pred = postEstimation.group_variables_for_time_series(grouping_variables = grouping_variables, df=df, frequency='quarterly')
income_series_true = postEstimation.group_variables_for_time_series(grouping_variables = grouping_variables, df=df_true, frequency='quarterly')
income_series_wb   = postEstimation.group_variables_for_time_series(grouping_variables = grouping_variables, df=df_wb, frequency='quarterly')

# Plotting:
plt.clf()
plt.figure(figsize=(10, 10))

# Plotting the means with standard deviation
plt.errorbar(income_series_true['date'], income_series_true['income_pc_weighted'], yerr=income_series_true['std_mean'], 
            label='True Income', color=settings.color1, fmt='-')
plt.errorbar(income_series_pred['date'], income_series_pred['income_pc_hat_weighted'], yerr=income_series_pred['std_hat_mean'], 
            label='Predicted Income (GB)', color=settings.color1, fmt='--', linestyle='--')  # Adjust linestyle if needed
plt.errorbar(income_series_wb['date'], income_series_wb['income_pc_hat_weighted'], yerr=income_series_wb['std_hat_mean'], 
            label='Predicted Income (WB)', color=settings.color1, fmt='--', linestyle=':')  # Adjust linestyle if needed

plt.xlabel('Date')
plt.ylabel('Income')
plt.legend(loc='lower left')
plt.savefig('../figures/fig7_average_income_time_series_quarterly.pdf', bbox_inches='tight')

print('Figure 7 saved')


#%% Figure 8 (fig8_poverty_rate_time_series): 
# Poverty Rate (Yearly)
#-------------------------------------------

grouping_variables = ['year']

income_series_pred = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df, frequency='yearly')
income_series_true = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df_true, frequency='yearly')
income_series_wb = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df_wb, frequency='yearly')

# Plotting:
plt.clf()
plt.figure(figsize=(10, 10))

# Plotting the means with standard deviation
# poor_685
plt.errorbar(income_series_true['date'], income_series_true['poor_685'], yerr=income_series_true['std_685_mean'], 
            label='LP 685', color=settings.color1, fmt='-')
plt.errorbar(income_series_pred['date'], income_series_pred['poor_hat_685'], yerr=income_series_pred['std_685_mean'], 
            label='LP 685 Predict', color=settings.color1, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb['date'], income_series_wb['poor_hat_685'], yerr=income_series_wb['std_685_mean'], 
            label='LP 685 Predict (WB)', color=settings.color1, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true['date'], income_series_true['poor_365'], yerr=income_series_true['std_365_mean'], 
            label='LP 365', color=settings.color3, fmt='-')
plt.errorbar(income_series_pred['date'], income_series_pred['poor_hat_365'], yerr=income_series_pred['std_365_mean'], 
            label='LP 365 Predict', color=settings.color3, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb['date'], income_series_wb['poor_hat_365'], yerr=income_series_wb['std_365_mean'], 
            label='LP 365 Predict (WB)', color=settings.color3, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true['date'], income_series_true['poor_215'], yerr=income_series_true['std_215_mean'], 
            label='LP 215', color=settings.color5, fmt='-')
plt.errorbar(income_series_pred['date'], income_series_pred['poor_hat_215'], yerr=income_series_pred['std_215_mean'], 
            label='LP 215 Predict', color=settings.color5, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb['date'], income_series_wb['poor_hat_215'], yerr=income_series_wb['std_215_mean'], 
            label='LP 215 Predict (WB)', color=settings.color5, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.xlabel('Date')
plt.ylabel('Poverty Rate')
plt.legend(loc='lower left')
plt.savefig('../figures/fig8_poverty_rate_time_series.pdf', bbox_inches='tight')

print('Figure 8 saved')


#%% Figure 8a (fig8a_poverty_rate_time_series_urbano): 
# Poverty Rate Urbano (Yearly)
#-------------------------------------------

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

# Plotting:
plt.clf()
plt.figure(figsize=(10, 10))
# Plotting the means with standard deviation
# poor_685
plt.errorbar(income_series_true_urban['date'], income_series_true_urban['poor_685'], yerr=income_series_true_urban['std_685_mean'], 
            label='LP 685', color=settings.color1, fmt='-')
plt.errorbar(income_series_pred_urban['date'], income_series_pred_urban['poor_hat_685'], yerr=income_series_pred_urban['std_685_mean'], 
            label='LP 685 Predict', color=settings.color1, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_urban['date'], income_series_wb_urban['poor_hat_685'], yerr=income_series_wb_urban['std_685_mean'], 
            label='LP 685 Predict (WB)', color=settings.color1, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true_urban['date'], income_series_true_urban['poor_365'], yerr=income_series_true_urban['std_365_mean'], 
            label='LP 365', color=settings.color3, fmt='-')
plt.errorbar(income_series_pred_urban['date'], income_series_pred_urban['poor_hat_365'], yerr=income_series_pred_urban['std_365_mean'], 
            label='LP 365 Predict', color=settings.color3, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_urban['date'], income_series_wb_urban['poor_hat_365'], yerr=income_series_wb_urban['std_365_mean'], 
            label='LP 365 Predict (WB)', color=settings.color3, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true_urban['date'], income_series_true_urban['poor_215'], yerr=income_series_true_urban['std_215_mean'], 
            label='LP 215', color=settings.color5, fmt='-')
plt.errorbar(income_series_pred_urban['date'], income_series_pred_urban['poor_hat_215'], yerr=income_series_pred_urban['std_215_mean'], 
            label='LP 215 Predict', color=settings.color5, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_urban['date'], income_series_wb_urban['poor_hat_215'], yerr=income_series_wb_urban['std_215_mean'], 
            label='LP 215 Predict (WB)', color=settings.color5, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.xlabel('Date')
plt.ylabel('Poverty Rate')
plt.legend(loc='lower left')
plt.savefig('../figures/fig8a_poverty_rate_time_series_urbano.pdf', bbox_inches='tight')

print('Figure 8a saved')


#%% Figure 8b (fig8a_poverty_rate_time_series_urbano): 
# Poverty Rate Rural (Yearly)
#-------------------------------------------

# Plotting:
plt.clf()
plt.figure(figsize=(10, 10))
# Plotting the means with standard deviation
# poor_685
plt.errorbar(income_series_true_rural['date'], income_series_true_rural['poor_685'], yerr=income_series_true_rural['std_685_mean'], 
            label='LP 685', color=settings.color1, fmt='-')
plt.errorbar(income_series_pred_rural['date'], income_series_pred_rural['poor_hat_685'], yerr=income_series_pred_rural['std_685_mean'], 
            label='LP 685 Predict', color=settings.color1, fmt='-.', linestyle='--')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_rural['date'], income_series_wb_rural['poor_hat_685'], yerr=income_series_wb_rural['std_685_mean'], 
            label='LP 685 Predict (WB)', color=settings.color1, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true_rural['date'], income_series_true_rural['poor_365'], yerr=income_series_true_rural['std_365_mean'], 
            label='LP 365', color=settings.color3, fmt='-')
plt.errorbar(income_series_pred_rural['date'], income_series_pred_rural['poor_hat_365'], yerr=income_series_pred_rural['std_365_mean'], 
            label='LP 365 Predict', color=settings.color3, fmt='-.', linestyle='--')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_rural['date'], income_series_wb_rural['poor_hat_365'], yerr=income_series_wb_rural['std_365_mean'], 
            label='LP 365 Predict (WB)', color=settings.color3, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true_rural['date'], income_series_true_rural['poor_215'], yerr=income_series_true_rural['std_215_mean'], 
            label='LP 215', color=settings.color5, fmt='-')
plt.errorbar(income_series_pred_rural['date'], income_series_pred_rural['poor_hat_215'], yerr=income_series_pred_rural['std_215_mean'], 
            label='LP 215 Predict', color=settings.color5, fmt='-.', linestyle='--')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_rural['date'], income_series_wb_rural['poor_hat_215'], yerr=income_series_wb_rural['std_215_mean'], 
            label='LP 215 Predict (WB)', color=settings.color5, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.xlabel('Date')
plt.ylabel('Poverty Rate')
plt.legend(loc='lower left')
plt.savefig('../figures/fig8b_poverty_rate_time_series_rural.pdf', bbox_inches='tight')

print('Figure 8b saved')

#%% Figure 8c (fig8a_poverty_rate_time_series_urbano): 
# Poverty Rate Urbano (Yearly)
#-------------------------------------------

grouping_variables = ['year']

income_series_pred = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df.query('lima_metropolitana==1'), frequency='yearly')
income_series_true = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df_true.query('lima_metropolitana==1'), frequency='yearly')
income_series_wb = postEstimation.group_porverty_rate_for_time_series(grouping_variables, df_wb.query('lima_metropolitana==1'), frequency='yearly')


# Plotting:
plt.clf()
plt.figure(figsize=(10, 10))
# Plotting the means with standard deviation
plt.errorbar(income_series_true['date'], income_series_true['poor_685'], yerr=income_series_true['std_685_mean'], 
            label='LP 685', color=settings.color1, fmt='-')
plt.errorbar(income_series_pred['date'], income_series_pred['poor_hat_685'], yerr=income_series_pred['std_685_mean'], 
            label='LP 685 Predict', color=settings.color1, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb['date'], income_series_wb['poor_hat_685'], yerr=income_series_wb['std_685_mean'], 
            label='LP 685 Predict (WB)', color=settings.color1, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true['date'], income_series_true['poor_365'], yerr=income_series_true['std_365_mean'], 
            label='LP 365', color=settings.color3, fmt='-')
plt.errorbar(income_series_pred['date'], income_series_pred['poor_hat_365'], yerr=income_series_pred['std_365_mean'], 
            label='LP 365 Predict', color=settings.color3, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb['date'], income_series_wb['poor_hat_365'], yerr=income_series_wb['std_365_mean'], 
            label='LP 365 Predict (WB)', color=settings.color3, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true['date'], income_series_true['poor_215'], yerr=income_series_true['std_215_mean'], 
            label='LP 215', color=settings.color5, fmt='-')
plt.errorbar(income_series_pred['date'], income_series_pred['poor_hat_215'], yerr=income_series_pred['std_215_mean'], 
            label='LP 215 Predict', color=settings.color5, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb['date'], income_series_wb['poor_hat_215'], yerr=income_series_wb['std_215_mean'], 
            label='LP 215 Predict (WB)', color=settings.color5, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.xlabel('Date')
plt.ylabel('Poverty Rate')
plt.legend(loc='lower left')
plt.savefig('../figures/fig8c_poverty_rate_time_series_lima.pdf', bbox_inches='tight')

print('Figure 8c saved')



#%% Figure 10a (fig10a_poverty_rate_time_series_male): 
# Poverty Rate Urbano (Yearly)
#-------------------------------------------

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

# Plotting:
plt.clf()
plt.figure(figsize=(10, 10))
# Plotting the means with standard deviation
# poor_685
plt.errorbar(income_series_true_male['date'], income_series_true_male['poor_685'], yerr=income_series_true_male['std_685_mean'], 
            label='LP 685', color=settings.color1, fmt='-')
plt.errorbar(income_series_pred_male['date'], income_series_pred_male['poor_hat_685'], yerr=income_series_pred_male['std_685_mean'], 
            label='LP 685 Predict', color=settings.color1, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_male['date'], income_series_wb_male['poor_hat_685'], yerr=income_series_wb_male['std_685_mean'], 
            label='LP 685 Predict (WB)', color=settings.color1, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true_male['date'], income_series_true_male['poor_365'], yerr=income_series_true_male['std_365_mean'], 
            label='LP 365', color=settings.color3, fmt='-')
plt.errorbar(income_series_pred_male['date'], income_series_pred_male['poor_hat_365'], yerr=income_series_pred_male['std_365_mean'], 
            label='LP 365 Predict', color=settings.color3, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_male['date'], income_series_wb_male['poor_hat_365'], yerr=income_series_wb_male['std_365_mean'], 
            label='LP 365 Predict (WB)', color=settings.color3, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true_male['date'], income_series_true_male['poor_215'], yerr=income_series_true_male['std_215_mean'], 
            label='LP 215', color=settings.color5, fmt='-')
plt.errorbar(income_series_pred_male['date'], income_series_pred_male['poor_hat_215'], yerr=income_series_pred_male['std_215_mean'], 
            label='LP 215 Predict', color=settings.color5, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_male['date'], income_series_wb_male['poor_hat_215'], yerr=income_series_wb_male['std_215_mean'], 
            label='LP 215 Predict (WB)', color=settings.color5, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.xlabel('Date')
plt.ylabel('Poverty Rate')
plt.legend(loc='lower left')
plt.savefig('../figures/fig10a_poverty_rate_time_series_male.pdf', bbox_inches='tight')

print('Figure 8a saved')


#%% Figure 10a (fig10b_poverty_rate_time_series_female): 
# Poverty Rate Rural (Yearly)
#-------------------------------------------

# Plotting:
plt.clf()
plt.figure(figsize=(10, 10))
# Plotting the means with standard deviation
# poor_685

plt.errorbar(income_series_true_female['date'], income_series_true_female['poor_685'], yerr=income_series_true_female['std_685_mean'], 
            label='LP 685', color=settings.color1, fmt='-')
plt.errorbar(income_series_pred_female['date'], income_series_pred_female['poor_hat_685'], yerr=income_series_pred_female['std_685_mean'], 
            label='LP 685 Predict', color=settings.color1, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_female['date'], income_series_wb_female['poor_hat_685'], yerr=income_series_wb_female['std_685_mean'], 
            label='LP 685 Predict (WB)', color=settings.color1, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true_female['date'], income_series_true_female['poor_365'], yerr=income_series_true_female['std_365_mean'], 
            label='LP 365', color=settings.color3, fmt='-')
plt.errorbar(income_series_pred_female['date'], income_series_pred_female['poor_hat_365'], yerr=income_series_pred_female['std_365_mean'], 
            label='LP 365 Predict', color=settings.color3, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_female['date'], income_series_wb_female['poor_hat_365'], yerr=income_series_wb_female['std_365_mean'], 
            label='LP 365 Predict (WB)', color=settings.color3, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true_female['date'], income_series_true_female['poor_215'], yerr=income_series_true_female['std_215_mean'], 
            label='LP 215', color=settings.color5, fmt='-')
plt.errorbar(income_series_pred_female['date'], income_series_pred_female['poor_hat_215'], yerr=income_series_pred_female['std_215_mean'], 
            label='LP 215 Predict', color=settings.color5, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_female['date'], income_series_wb_female['poor_hat_215'], yerr=income_series_wb_female['std_215_mean'], 
            label='LP 215 Predict (WB)', color=settings.color5, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.xlabel('Date')
plt.ylabel('Poverty Rate')
plt.legend(loc='lower left')
plt.savefig('../figures/fig10b_poverty_rate_time_series_female.pdf', bbox_inches='tight')

print('Figure 8b saved')


#%% Figure 11a (fig11a_poverty_rate_time_series_formal): 'Informal', 'Not Informal'
# Poverty Rate Urbano (Yearly)
#-------------------------------------------

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

# Plotting:
plt.clf()
plt.figure(figsize=(10, 10))
# Plotting the means with standard deviation
# poor_685
plt.errorbar(income_series_true_informal['date'], income_series_true_informal['poor_685'], yerr=income_series_true_informal['std_685_mean'], 
            label='LP 685', color=settings.color1, fmt='-')
plt.errorbar(income_series_pred_informal['date'], income_series_pred_informal['poor_hat_685'], yerr=income_series_pred_informal['std_685_mean'], 
            label='LP 685 Predict', color=settings.color1, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_informal['date'], income_series_wb_informal['poor_hat_685'], yerr=income_series_wb_informal['std_685_mean'], 
            label='LP 685 Predict (WB)', color=settings.color1, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true_informal['date'], income_series_true_informal['poor_365'], yerr=income_series_true_informal['std_365_mean'], 
            label='LP 365', color=settings.color3, fmt='-')
plt.errorbar(income_series_pred_informal['date'], income_series_pred_informal['poor_hat_365'], yerr=income_series_pred_informal['std_365_mean'], 
            label='LP 365 Predict', color=settings.color3, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_informal['date'], income_series_wb_informal['poor_hat_365'], yerr=income_series_wb_informal['std_365_mean'], 
            label='LP 365 Predict (WB)', color=settings.color3, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true_informal['date'], income_series_true_informal['poor_215'], yerr=income_series_true_informal['std_215_mean'], 
            label='LP 215', color=settings.color5, fmt='-')
plt.errorbar(income_series_pred_informal['date'], income_series_pred_informal['poor_hat_215'], yerr=income_series_pred_informal['std_215_mean'], 
            label='LP 215 Predict', color=settings.color5, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_informal['date'], income_series_wb_informal['poor_hat_215'], yerr=income_series_wb_informal['std_215_mean'], 
            label='LP 215 Predict (WB)', color=settings.color5, fmt='-.', linestyle=':')  # Adjust linestyle if needed
plt.xlabel('Date')
plt.ylabel('Poverty Rate')
plt.legend(loc='lower left')
plt.savefig('../figures/fig10a_poverty_rate_time_series_informal.pdf', bbox_inches='tight')

print('Figure 8a saved')


#%% Figure 11b (fig11b_poverty_rate_time_series_informal): 
# Poverty Rate Rural (Yearly)
#-------------------------------------------

# Plotting:
plt.clf()
plt.figure(figsize=(10, 10))
# Plotting the means with standard deviation
# poor_685
plt.errorbar(income_series_true_formal['date'], income_series_true_formal['poor_685'], yerr=income_series_true_formal['std_685_mean'], 
            label='LP 685', color=settings.color1, fmt='-')
plt.errorbar(income_series_pred_formal['date'], income_series_pred_formal['poor_hat_685'], yerr=income_series_pred_formal['std_685_mean'], 
            label='LP 685 Predict', color=settings.color1, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_formal['date'], income_series_wb_formal['poor_hat_685'], yerr=income_series_wb_formal['std_685_mean'], 
            label='LP 685 Predict (WB)', color=settings.color1, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true_formal['date'], income_series_true_formal['poor_365'], yerr=income_series_true_formal['std_365_mean'], 
            label='LP 365', color=settings.color3, fmt='-')
plt.errorbar(income_series_pred_formal['date'], income_series_pred_formal['poor_hat_365'], yerr=income_series_pred_formal['std_365_mean'], 
            label='LP 365 Predict', color=settings.color3, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_pred_formal['date'], income_series_pred_formal['poor_hat_365'], yerr=income_series_pred_formal['std_365_mean'], 
            label='LP 365 Predict (WB)', color=settings.color3, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true_formal['date'], income_series_true_formal['poor_215'], yerr=income_series_true_formal['std_215_mean'], 
            label='LP 215', color=settings.color5, fmt='-')
plt.errorbar(income_series_pred_formal['date'], income_series_pred_formal['poor_hat_215'], yerr=income_series_pred_formal['std_215_mean'], 
            label='LP 215 Predict', color=settings.color5, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_formal['date'], income_series_wb_formal['poor_hat_215'], yerr=income_series_wb_formal['std_215_mean'], 
            label='LP 215 Predict (WB)', color=settings.color5, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.xlabel('Date')
plt.ylabel('Poverty Rate')
plt.legend(loc='lower left')
plt.savefig('../figures/fig10b_poverty_rate_time_series_formal.pdf', bbox_inches='tight')

print('Figure 8b saved')


#%% Figure 12a (fig12a_poverty_rate_time_series_educ): 'Elementary', 'Superior'
# Poverty Rate Urbano (Yearly)
#-------------------------------------------

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

# Plotting:
plt.clf()
plt.figure(figsize=(10, 10))
# Plotting the means with standard deviation
# poor_685
plt.errorbar(income_series_true_elementary['date'], income_series_true_elementary['poor_685'], yerr=income_series_true_elementary['std_685_mean'], 
            label='LP 685', color=settings.color1, fmt='-')
plt.errorbar(income_series_pred_elementary['date'], income_series_pred_elementary['poor_hat_685'], yerr=income_series_pred_elementary['std_685_mean'], 
            label='LP 685 Predict', color=settings.color1, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_elementary['date'], income_series_wb_elementary['poor_hat_685'], yerr=income_series_wb_elementary['std_685_mean'], 
            label='LP 685 Predict (WB)', color=settings.color1, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true_elementary['date'], income_series_true_elementary['poor_365'], yerr=income_series_true_elementary['std_365_mean'], 
            label='LP 365', color=settings.color3, fmt='-')
plt.errorbar(income_series_pred_elementary['date'], income_series_pred_elementary['poor_hat_365'], yerr=income_series_pred_elementary['std_365_mean'], 
            label='LP 365 Predict', color=settings.color3, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_elementary['date'], income_series_wb_elementary['poor_hat_365'], yerr=income_series_wb_elementary['std_365_mean'], 
            label='LP 365 Predict (WB)', color=settings.color3, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true_elementary['date'], income_series_true_elementary['poor_215'], yerr=income_series_true_elementary['std_215_mean'], 
            label='LP 215', color=settings.color5, fmt='-')
plt.errorbar(income_series_pred_elementary['date'], income_series_pred_elementary['poor_hat_215'], yerr=income_series_pred_elementary['std_215_mean'], 
            label='LP 215 Predict', color=settings.color5, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_elementary['date'], income_series_wb_elementary['poor_hat_215'], yerr=income_series_wb_elementary['std_215_mean'], 
            label='LP 215 Predict (WB)', color=settings.color5, fmt='-.', linestyle=':')  # Adjust linestyle if needed
plt.xlabel('Date')
plt.ylabel('Poverty Rate')
plt.legend(loc='lower left')
plt.savefig('../figures/fig12a_poverty_rate_time_series_elementary.pdf', bbox_inches='tight')

print('Figure 12a saved')


#%% Figure 11b (fig11b_poverty_rate_time_series_informal): 
# Poverty Rate Rural (Yearly)
#-------------------------------------------

# Plotting:
plt.clf()
plt.figure(figsize=(10, 10))
# Plotting the means with standard deviation
# poor_685
plt.errorbar(income_series_true_superior['date'], income_series_true_superior['poor_685'], yerr=income_series_true_superior['std_685_mean'], 
            label='LP 685', color=settings.color1, fmt='-')
plt.errorbar(income_series_pred_superior['date'], income_series_pred_superior['poor_hat_685'], yerr=income_series_pred_superior['std_685_mean'], 
            label='LP 685 Predict', color=settings.color1, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_superior['date'], income_series_wb_superior['poor_hat_685'], yerr=income_series_wb_superior['std_685_mean'], 
            label='LP 685 Predict (WB)', color=settings.color1, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true_superior['date'], income_series_true_superior['poor_365'], yerr=income_series_true_superior['std_365_mean'], 
            label='LP 365', color=settings.color3, fmt='-')
plt.errorbar(income_series_pred_superior['date'], income_series_pred_superior['poor_hat_365'], yerr=income_series_pred_superior['std_365_mean'], 
            label='LP 365 Predict', color=settings.color3, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_pred_superior['date'], income_series_pred_superior['poor_hat_365'], yerr=income_series_pred_superior['std_365_mean'], 
            label='LP 365 Predict (WB)', color=settings.color3, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true_superior['date'], income_series_true_superior['poor_215'], yerr=income_series_true_superior['std_215_mean'], 
            label='LP 215', color=settings.color5, fmt='-')
plt.errorbar(income_series_pred_superior['date'], income_series_pred_superior['poor_hat_215'], yerr=income_series_pred_superior['std_215_mean'], 
            label='LP 215 Predict', color=settings.color5, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_superior['date'], income_series_wb_superior['poor_hat_215'], yerr=income_series_wb_superior['std_215_mean'], 
            label='LP 215 Predict (WB)', color=settings.color5, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.xlabel('Date')
plt.ylabel('Poverty Rate')
plt.legend(loc='lower left')
plt.savefig('../figures/fig12b_poverty_rate_time_series_superior.pdf', bbox_inches='tight')

print('Figure 8b saved')



#%% Figure 13a (fig13a_poverty_rate_time_series_nchildren): 'Elementary', 'Superior'
# Poverty Rate Urbano (Yearly)
#-------------------------------------------

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


# Plotting:
plt.clf()
plt.figure(figsize=(10, 10))
# Plotting the means with standard deviation
# poor_685

plt.errorbar(income_series_true_nchild0['date'], income_series_true_nchild0['poor_685'], yerr=income_series_true_nchild0['std_685_mean'], 
            label='0 Child', color=settings.color1, fmt='-')
plt.errorbar(income_series_pred_nchild0['date'], income_series_pred_nchild0['poor_hat_685'], yerr=income_series_pred_nchild0['std_685_mean'], 
            label='0 Child Predict', color=settings.color1, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_nchild0['date'], income_series_wb_nchild0['poor_hat_685'], yerr=income_series_wb_nchild0['std_685_mean'], 
            label='0 Child Predict (WB)', color=settings.color1, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true_nchild1['date'], income_series_true_nchild1['poor_685'], yerr=income_series_true_nchild1['std_685_mean'], 
            label='1 Child', color=settings.color2, fmt='-')
plt.errorbar(income_series_pred_nchild1['date'], income_series_pred_nchild1['poor_hat_685'], yerr=income_series_pred_nchild1['std_685_mean'], 
            label='1 Child Predict', color=settings.color2, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_nchild1['date'], income_series_wb_nchild1['poor_hat_685'], yerr=income_series_wb_nchild1['std_685_mean'], 
            label='1 Child Predict (WB)', color=settings.color2, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true_nchild2['date'], income_series_true_nchild2['poor_685'], yerr=income_series_true_nchild2['std_685_mean'], 
            label='2 Child', color=settings.color3, fmt='-')
plt.errorbar(income_series_pred_nchild2['date'], income_series_pred_nchild2['poor_hat_685'], yerr=income_series_pred_nchild2['std_685_mean'], 
            label='2 Child Predict', color=settings.color3, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_nchild2['date'], income_series_wb_nchild2['poor_hat_685'], yerr=income_series_wb_nchild2['std_685_mean'], 
            label='2 Child Predict (WB)', color=settings.color3, fmt='-.', linestyle=':')  # Adjust linestyle if needed

plt.errorbar(income_series_true_nchild3['date'], income_series_true_nchild3['poor_685'], yerr=income_series_true_nchild3['std_685_mean'], 
            label='3 Child', color=settings.color4, fmt='-')
plt.errorbar(income_series_pred_nchild3['date'], income_series_pred_nchild3['poor_hat_685'], yerr=income_series_pred_nchild3['std_685_mean'], 
            label='3 Child Predict', color=settings.color4, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_wb_nchild3['date'], income_series_wb_nchild3['poor_hat_685'], yerr=income_series_wb_nchild3['std_685_mean'], 
            label='3 Child Predict (WB)', color=settings.color4, fmt='-.', linestyle=':')  # Adjust linestyle if needed


plt.xlabel('Date')
plt.ylabel('Poverty Rate')
plt.legend(loc='lower left')
plt.savefig('../figures/fig12a_poverty_rate_time_series_nchild.pdf', bbox_inches='tight')

print('Figure 12a saved')





#%% Figure 9 (fig9_gini_time_series): 
# Gini Coefficient
#------------------------------------------

household_weight = df['n_people']/df.groupby('year')['n_people'].transform('sum')

def gini_coefficient(income_data):
    sorted_income = np.sort(income_data)
    n = len(income_data)
    cumulative_income = np.cumsum(sorted_income)
    gini = (n + 1 - 2 * np.sum(cumulative_income) / cumulative_income[-1]) / n
    return gini

income_series_pred = df.groupby('year')['income_pc_hat'].apply(gini_coefficient).reset_index().rename(columns={'income_pc_hat':'gini'})

income_series_true = df_true.groupby('year')['income_pc'].apply(gini_coefficient).reset_index().rename(columns={'income_pc':'gini'})


# Convert 'year' and 'month' to a datetime
income_series_pred['date'] = pd.to_datetime(income_series_pred[['year']].assign(MONTH=1,DAY=1))
income_series_true['date'] = pd.to_datetime(income_series_true[['year']].assign(MONTH=1,DAY=1))

# Plotting:
plt.clf()
# Plotting the means with standard deviation
plt.figure(figsize=(10, 10))
plt.plot(income_series_true['date'], income_series_true['gini'], label='True GINI', color=settings.color2)
plt.plot(income_series_pred['date'], income_series_pred['gini'], label='Predicted GINI', color=settings.color4, linestyle='--')
plt.xlabel('Date')
plt.ylabel('Income')
plt.ylim(0.2, .8)
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig9_gini_time_series.pdf', bbox_inches='tight')

print('Figure 9 saved')


print('End of code: 04_generate_prediction_report.py')






