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

date = '2024-04-24' #datetime.today().strftime('%Y-%m-%d')

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
                                    ).reset_index(drop=True)
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
ml_dataset_filtered_validation_world_bank = postEstimation.compute_predicted_income_world_bank(ml_dataset_filtered_validation_world_bank, forecast='predicted')
ml_dataset_filtered_validation_world_bank = postEstimation.compute_predicted_income_world_bank(ml_dataset_filtered_validation_world_bank, forecast='perfect')
ml_dataset_filtered_validation_world_bank = postEstimation.compute_predicted_income_world_bank(ml_dataset_filtered_validation_world_bank, forecast='lagged')

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
df.loc[df['year'] <= base_year, 'income_pc_hat'] = df.loc[df['year'] <= base_year, 'income_pc'] # Change income_pc_hat to income_pc for years <= 2016

df_wb.loc[df_wb['year'] <= base_year, 'income_pc_hat'] = df_wb.loc[df_wb['year'] <= base_year, 'income_pc'] # Change income_pc_hat to income_pc for years <= 2016



#%% Plot with standard deviation:

plt.clf()
plt.plot(df.groupby('year').predicted_error.std())
plt.ylim([0,1])
plt.savefig('../figures/std_trend.pdf', bbox_inches='tight')

#%% Figure 0 (Binned scatterplot:)
#----------------------------------------------------------------

# Run scatter plot with distribution for prediction year (default = 2016)
df_pred_year = ml_dataset_filtered_train.query('year >=' + str(base_year+1)).query('year <= ' + str(base_year+3)).copy()

figuresReport.create_binned_scatterplot(
    df=df_pred_year.copy(), 
    income_col='log_income_pc', 
    predicted_col='log_income_pc_hat', 
    bin_width=0.1, 
    bin_start=2, 
    bin_end=df_pred_year['log_income_pc'].max()
    )

#%% Figure 1 (fig1_prediction_vs_true_income_distribution_lasso_training_weighted): 
# Distribution of predicted income vs true income
#--------------------------------------------------------------

plt.clf()
plt.figure(figsize=(10, 10))
sns.histplot(ml_dataset_filtered_validation['income_pc_hat'], 
                color=settings.color1, 
            #     kde=True, 
                label='Predicted Income', 
                stat='density', 
                fill=False, 
                element='step'
                )
sns.histplot(ml_dataset_filtered_true['income_pc'], 
                color=settings.color2, 
            #     kde=True, 
                label='True Income', 
                stat='density', 
                fill=False, 
                element='step'
                )
plt.xlim(0, 3000)
plt.legend()
plt.savefig('../figures/fig1_prediction_vs_true_income_distribution_lasso_training_weighted.pdf', bbox_inches='tight')
print('Figure 1 saved')



#%% Figure 1b (fig1b_prediction_vs_true_income_ecdf_lasso_training_weighted): 
# ECDF of predicted income vs true income
#-------------------------------------------------------

plt.clf()
plt.figure(figsize=(10, 10))
sns.ecdfplot(ml_dataset_filtered_validation['income_pc_hat'], color=settings.color1, label='Predicted Income')
sns.ecdfplot(ml_dataset_filtered_true['income_pc'], color=settings.color2, label='True Income')
plt.xlim(0, 2500)
plt.legend()
plt.xlabel('Income')
plt.ylabel('Cumulative Distribution')
plt.savefig('../figures/fig1b_prediction_vs_true_income_ecdf_lasso_training_weighted.pdf', bbox_inches='tight')
print('Figure 1b saved')


# %% Figure 1c (fig1c_prediction_vs_true_income_by_region_lasso_training_weighted):
#---------------------------------------------------------------------------------

plt.clf()
plt.figure(figsize=(10, 10))
sns.histplot(ml_dataset_filtered_validation['log_income_pc_hat'] , 
                color=settings.color1, 
            #     kde=True, 
                label='Predicted Income', 
                stat='density', 
                fill=False, 
                element='step'
                )
sns.histplot(ml_dataset_filtered_true['log_income_pc'], 
                color=settings.color2, 
            #     kde=True, 
                label='True Income', 
                stat='density', 
                fill=False, 
                element='step'
                )
# plt.xlim(0, 3000)
plt.legend()
plt.savefig('../figures/fig1c_prediction_vs_true_income_distribution_lasso_training_weighted.pdf', bbox_inches='tight')
print('Figure 1c saved')



#%% Figure 2 (fig2_prediction_vs_true_income_by_region_lasso_training_weighted): 
# Distribution of predicted income vs true income by region
#---------------------------------------------------------------------------------

plt.clf()

n_rows = 5
n_cols = 5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20), sharex=True, sharey=True)

for i, region in enumerate(ml_dataset_filtered_validation['ubigeo_region'].unique()):
    ax = axes[i // n_cols, i % n_cols]
    region_data = ml_dataset_filtered_validation[ml_dataset_filtered_validation['ubigeo_region'] == region]
    region_data_true = ml_dataset_filtered_true[ml_dataset_filtered_true['ubigeo_region'] == region]
    sns.histplot(region_data['income_pc_hat'], 
                    color=settings.color1, 
                    label='Gradient Boosting', 
                    stat='density', 
                    fill=False, 
                    element='step',
                    linewidth=2,  # Set the line width here
                    ax=ax)
    sns.histplot(region_data_true['income_pc'], 
                    color=settings.color2, 
                    label='True Income', 
                    stat='density', 
                    fill=False, 
                    element='step',
                    linewidth=2,  # Set the line width here
                    ax=ax)

    ax.set_xlim(0, 2000)
    ax.set_title(region)
    ax.legend()
    plt.savefig('../figures/fig2_prediction_vs_true_income_by_region_lasso_training_weighted.pdf', bbox_inches='tight')

print('Figure 2 saved')



#%% Figure 3 (fig3_prediction_vs_true_poverty_rate_national): 
# Poverty Rate National 
#-----------------------------------------------------------------------------------

# True data: (2017-2019)
ml_dataset_filtered_true['n_people'] = ml_dataset_filtered_true['mieperho'] * ml_dataset_filtered_true['pondera_i']
household_weight_test = ml_dataset_filtered_true['n_people']/ml_dataset_filtered_true.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_true['poor_685'] = (ml_dataset_filtered_true['income_pc'] <= ml_dataset_filtered_true['lp_685usd_ppp']) * household_weight_test
ml_dataset_filtered_true['poor_365'] = (ml_dataset_filtered_true['income_pc'] <= ml_dataset_filtered_true['lp_365usd_ppp']) * household_weight_test
ml_dataset_filtered_true['poor_215'] = (ml_dataset_filtered_true['income_pc'] <= ml_dataset_filtered_true['lp_215usd_ppp']) * household_weight_test

# Predicted data: (using 2016 data)
ml_dataset_filtered_validation['n_people'] = ml_dataset_filtered_validation['mieperho'] * ml_dataset_filtered_validation['pondera_i']
household_weight_prediction = ml_dataset_filtered_validation['n_people']/ml_dataset_filtered_validation.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_validation['poor_hat_685'] = (ml_dataset_filtered_validation['income_pc_hat'] <= ml_dataset_filtered_validation['lp_685usd_ppp']) * household_weight_prediction
ml_dataset_filtered_validation['poor_hat_365'] = (ml_dataset_filtered_validation['income_pc_hat'] <= ml_dataset_filtered_validation['lp_365usd_ppp']) * household_weight_prediction
ml_dataset_filtered_validation['poor_hat_215'] = (ml_dataset_filtered_validation['income_pc_hat'] <= ml_dataset_filtered_validation['lp_215usd_ppp']) * household_weight_prediction

# Get difference between the true and predicted national rate:
porverty_comparison_test = ml_dataset_filtered_true.loc[:,['year','poor_685','poor_365','poor_215']].groupby('year').sum()
porverty_comparison_pred = ml_dataset_filtered_validation.loc[:,['year','poor_hat_685','poor_hat_365','poor_hat_215']].groupby('year').sum()
porverty_comparison_diff = porverty_comparison_test.copy()
porverty_comparison_diff.iloc[:,:] = np.array(porverty_comparison_test) - np.array(porverty_comparison_pred)


# Plotting
plt.clf()
ax = np.abs(porverty_comparison_diff).plot.bar(figsize=(10, 6), width=0.8)
ax.set_xlabel('Poverty Threshold')
ax.set_ylabel('Difference: True - Predicted')
ax.set_title('Poverty Comparison by Year')
ax.set_xticklabels(porverty_comparison_diff.index, rotation=45)
plt.legend(title='Year',  loc='upper right')
plt.ylim([-.05, .2])
plt.tight_layout()
plt.grid(True)
# plt.show()
plt.savefig('../figures/fig3_prediction_vs_true_poverty_rate_national.pdf', bbox_inches='tight')

print('Figure 3 saved')


#%% Figure 3 (fig3_prediction_vs_true_poverty_rate_national): 
# Poverty Rate National World Bank
#-----------------------------------------------------------------------------------

# True data: (2017-2019)
ml_dataset_filtered_true['n_people'] = ml_dataset_filtered_true['mieperho'] * ml_dataset_filtered_true['pondera_i']
household_weight_test = ml_dataset_filtered_true['n_people']/ml_dataset_filtered_true.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_true['poor_685'] = (ml_dataset_filtered_true['income_pc'] <= ml_dataset_filtered_true['lp_685usd_ppp']) * household_weight_test
ml_dataset_filtered_true['poor_365'] = (ml_dataset_filtered_true['income_pc'] <= ml_dataset_filtered_true['lp_365usd_ppp']) * household_weight_test
ml_dataset_filtered_true['poor_215'] = (ml_dataset_filtered_true['income_pc'] <= ml_dataset_filtered_true['lp_215usd_ppp']) * household_weight_test

# Predicted data: (using 2016 data)
ml_dataset_filtered_validation['n_people'] = ml_dataset_filtered_validation['mieperho'] * ml_dataset_filtered_validation['pondera_i']
household_weight_prediction = ml_dataset_filtered_validation['n_people']/ml_dataset_filtered_validation.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_validation['poor_hat_685'] = (ml_dataset_filtered_validation['income_pc_hat'] <= ml_dataset_filtered_validation['lp_685usd_ppp']) * household_weight_prediction
ml_dataset_filtered_validation['poor_hat_365'] = (ml_dataset_filtered_validation['income_pc_hat'] <= ml_dataset_filtered_validation['lp_365usd_ppp']) * household_weight_prediction
ml_dataset_filtered_validation['poor_hat_215'] = (ml_dataset_filtered_validation['income_pc_hat'] <= ml_dataset_filtered_validation['lp_215usd_ppp']) * household_weight_prediction

# Predicted data WB: (using 2016 data)
ml_dataset_filtered_validation_world_bank['n_people'] = ml_dataset_filtered_validation_world_bank['mieperho'] * ml_dataset_filtered_validation_world_bank['pondera_i']
household_weight_prediction = ml_dataset_filtered_validation_world_bank['n_people']/ml_dataset_filtered_validation_world_bank.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_validation_world_bank['poor_hat_685'] = (ml_dataset_filtered_validation_world_bank['income_pc_hat'] <= ml_dataset_filtered_validation_world_bank['lp_685usd_ppp']) * household_weight_prediction
ml_dataset_filtered_validation_world_bank['poor_hat_365'] = (ml_dataset_filtered_validation_world_bank['income_pc_hat'] <= ml_dataset_filtered_validation_world_bank['lp_365usd_ppp']) * household_weight_prediction
ml_dataset_filtered_validation_world_bank['poor_hat_215'] = (ml_dataset_filtered_validation_world_bank['income_pc_hat'] <= ml_dataset_filtered_validation_world_bank['lp_215usd_ppp']) * household_weight_prediction

# Predicted data WB Perfect Forecast: (using 2016 data)
ml_dataset_filtered_validation_world_bank['n_people'] = ml_dataset_filtered_validation_world_bank['mieperho'] * ml_dataset_filtered_validation_world_bank['pondera_i']
household_weight_prediction = ml_dataset_filtered_validation_world_bank['n_people']/ml_dataset_filtered_validation_world_bank.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_validation_world_bank['poor_hat_pf_685'] = (ml_dataset_filtered_validation_world_bank['income_pc_hat_perfect_forecast'] <= ml_dataset_filtered_validation_world_bank['lp_685usd_ppp']) * household_weight_prediction
ml_dataset_filtered_validation_world_bank['poor_hat_pf_365'] = (ml_dataset_filtered_validation_world_bank['income_pc_hat_perfect_forecast'] <= ml_dataset_filtered_validation_world_bank['lp_365usd_ppp']) * household_weight_prediction
ml_dataset_filtered_validation_world_bank['poor_hat_pf_215'] = (ml_dataset_filtered_validation_world_bank['income_pc_hat_perfect_forecast'] <= ml_dataset_filtered_validation_world_bank['lp_215usd_ppp']) * household_weight_prediction

# Predicted data WB Forecast Oct. t-1: (using 2016 data)
ml_dataset_filtered_validation_world_bank['n_people'] = ml_dataset_filtered_validation_world_bank['mieperho'] * ml_dataset_filtered_validation_world_bank['pondera_i']
household_weight_prediction = ml_dataset_filtered_validation_world_bank['n_people']/ml_dataset_filtered_validation_world_bank.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_validation_world_bank['poor_hat_lf_685'] = (ml_dataset_filtered_validation_world_bank['income_pc_hat_lagged_forecast'] <= ml_dataset_filtered_validation_world_bank['lp_685usd_ppp']) * household_weight_prediction
ml_dataset_filtered_validation_world_bank['poor_hat_lf_365'] = (ml_dataset_filtered_validation_world_bank['income_pc_hat_lagged_forecast'] <= ml_dataset_filtered_validation_world_bank['lp_365usd_ppp']) * household_weight_prediction
ml_dataset_filtered_validation_world_bank['poor_hat_lf_215'] = (ml_dataset_filtered_validation_world_bank['income_pc_hat_lagged_forecast'] <= ml_dataset_filtered_validation_world_bank['lp_215usd_ppp']) * household_weight_prediction


yy = 2020

# Get difference between the true and predicted national rate:
porverty_comparison_test = ml_dataset_filtered_true.loc[:,['year','poor_685','poor_365','poor_215']].groupby('year').sum().loc[yy,:]
porverty_comparison_pred = ml_dataset_filtered_validation.loc[:,['year','poor_hat_685','poor_hat_365','poor_hat_215']].groupby('year').sum().loc[yy,:]
porverty_comparison_pred_wb = ml_dataset_filtered_validation_world_bank.loc[:,['year','poor_hat_685','poor_hat_365','poor_hat_215']].groupby('year').sum().loc[yy,:]
porverty_comparison_pred_wb_pf = ml_dataset_filtered_validation_world_bank.loc[:,['year','poor_hat_pf_685','poor_hat_pf_365','poor_hat_pf_215']].groupby('year').sum().loc[yy,:]
porverty_comparison_pred_wb_lf = ml_dataset_filtered_validation_world_bank.loc[:,['year','poor_hat_lf_685','poor_hat_lf_365','poor_hat_lf_215']].groupby('year').sum().loc[yy,:]
porverty_comparison_diff = pd.DataFrame(porverty_comparison_test.copy())

# Get difference between the true and predicted national rate:
porverty_comparison_diff.loc[:,'Forecast Oct. t-1'] = np.array(porverty_comparison_test) - np.array(porverty_comparison_pred_wb_lf)
porverty_comparison_diff.loc[:,'Gradient Boosting'] = np.array(porverty_comparison_test) - np.array(porverty_comparison_pred)
porverty_comparison_diff.loc[:,'Forecast Apr. t (WB)'] = np.array(porverty_comparison_test) - np.array(porverty_comparison_pred_wb)
porverty_comparison_diff.loc[:,'Perfect Forecast'] = np.array(porverty_comparison_test) - np.array(porverty_comparison_pred_wb_pf)

porverty_comparison_diff = porverty_comparison_diff.drop(columns=[yy])

# Plotting
plt.clf()
ax = np.abs(porverty_comparison_diff).plot.bar(figsize=(10, 6), width=0.8)
ax.set_xlabel('Poverty Threshold')
ax.set_ylabel('Difference: True - Predicted')
ax.set_title('Poverty Comparison by Year')
ax.set_xticklabels(porverty_comparison_diff.index, rotation=45)
plt.legend(title='Year',  loc='upper right')
plt.ylim([-.05, .2])
plt.tight_layout()
plt.grid(True)
# plt.show()
plt.savefig('../figures/fig3_poverty_prediction_' + str(yy) + '.pdf', bbox_inches='tight')

print('Figure 3 saved')

# Year
yy = 2021

# Get difference between the true and predicted national rate:
porverty_comparison_test = ml_dataset_filtered_true.loc[:,['year','poor_685','poor_365','poor_215']].groupby('year').sum().loc[yy,:]
porverty_comparison_pred = ml_dataset_filtered_validation.loc[:,['year','poor_hat_685','poor_hat_365','poor_hat_215']].groupby('year').sum().loc[yy,:]
porverty_comparison_pred_wb = ml_dataset_filtered_validation_world_bank.loc[:,['year','poor_hat_685','poor_hat_365','poor_hat_215']].groupby('year').sum().loc[yy,:]
porverty_comparison_pred_wb_pf = ml_dataset_filtered_validation_world_bank.loc[:,['year','poor_hat_pf_685','poor_hat_pf_365','poor_hat_pf_215']].groupby('year').sum().loc[yy,:]
porverty_comparison_pred_wb_lf = ml_dataset_filtered_validation_world_bank.loc[:,['year','poor_hat_lf_685','poor_hat_lf_365','poor_hat_lf_215']].groupby('year').sum().loc[yy,:]
porverty_comparison_diff = pd.DataFrame(porverty_comparison_test.copy())

# Get difference between the true and predicted national rate:
porverty_comparison_diff.loc[:,'Forecast Oct. t-1'] = np.array(porverty_comparison_test) - np.array(porverty_comparison_pred_wb_lf)
porverty_comparison_diff.loc[:,'Gradient Boosting'] = np.array(porverty_comparison_test) - np.array(porverty_comparison_pred)
porverty_comparison_diff.loc[:,'Forecast Apr. t (WB)'] = np.array(porverty_comparison_test) - np.array(porverty_comparison_pred_wb)
porverty_comparison_diff.loc[:,'Perfect Forecast'] = np.array(porverty_comparison_test) - np.array(porverty_comparison_pred_wb_pf)

porverty_comparison_diff = porverty_comparison_diff.drop(columns=[yy])

# Plotting
plt.clf()
ax = np.abs(porverty_comparison_diff).plot.bar(figsize=(10, 6), width=0.8)
ax.set_xlabel('Poverty Threshold')
ax.set_ylabel('Difference: True - Predicted')
ax.set_title('Poverty Comparison by Year')
ax.set_xticklabels(porverty_comparison_diff.index, rotation=45)
plt.legend(title='Year',  loc='upper right')
plt.ylim([-.05, .2])
plt.tight_layout()
plt.grid(True)
# plt.show()
plt.savefig('../figures/fig3_poverty_prediction_' + str(yy) + '.pdf', bbox_inches='tight')

print('Figure 3 saved')


#%% Figure 3 (fig3_prediction_vs_true_avg_income_national): 
# Poverty Rate National Forecast Apr. t
#-----------------------------------------------------------------------------------

# True data: (2017-2019)
ml_dataset_filtered_true['n_people'] = ml_dataset_filtered_true['mieperho'] * ml_dataset_filtered_true['pondera_i']
household_weight_test = ml_dataset_filtered_true['n_people']/ml_dataset_filtered_true.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_true['income_hat'] = (ml_dataset_filtered_true['income_pc'] ) * household_weight_test

# Predicted data: (using 2016 data)
ml_dataset_filtered_validation['n_people'] = ml_dataset_filtered_validation['mieperho'] * ml_dataset_filtered_validation['pondera_i']
household_weight_prediction = ml_dataset_filtered_validation['n_people']/ml_dataset_filtered_validation.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_validation['income_hat'] = (ml_dataset_filtered_validation['income_pc_hat'] ) * household_weight_prediction

# Predicted data WB: (using 2016 data)
ml_dataset_filtered_validation_world_bank['n_people'] = ml_dataset_filtered_validation_world_bank['mieperho'] * ml_dataset_filtered_validation_world_bank['pondera_i']
household_weight_prediction = ml_dataset_filtered_validation_world_bank['n_people']/ml_dataset_filtered_validation_world_bank.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_validation_world_bank['income_hat'] = (ml_dataset_filtered_validation_world_bank['income_pc_hat']) * household_weight_prediction

# Predicted data WB Perfect Forecast: (using 2016 data)
ml_dataset_filtered_validation_world_bank['n_people'] = ml_dataset_filtered_validation_world_bank['mieperho'] * ml_dataset_filtered_validation_world_bank['pondera_i']
household_weight_prediction = ml_dataset_filtered_validation_world_bank['n_people']/ml_dataset_filtered_validation_world_bank.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_validation_world_bank['income_hat_pf'] = (ml_dataset_filtered_validation_world_bank['income_pc_hat_perfect_forecast']) * household_weight_prediction

# Predicted data WB Forecast Oct. t-1: (using 2016 data)
ml_dataset_filtered_validation_world_bank['n_people'] = ml_dataset_filtered_validation_world_bank['mieperho'] * ml_dataset_filtered_validation_world_bank['pondera_i']
household_weight_prediction = ml_dataset_filtered_validation_world_bank['n_people']/ml_dataset_filtered_validation_world_bank.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_validation_world_bank['income_hat_lf'] = (ml_dataset_filtered_validation_world_bank['income_pc_hat_lagged_forecast'] ) * household_weight_prediction


yy = 2020

# Get difference between the true and predicted national rate:
porverty_comparison_test = ml_dataset_filtered_true.loc[:,['year','income_hat']].groupby('year').sum().loc[yy,:]
porverty_comparison_pred = ml_dataset_filtered_validation.loc[:,['year','income_hat']].groupby('year').sum().loc[yy,:]
porverty_comparison_pred_wb = ml_dataset_filtered_validation_world_bank.loc[:,['year','income_hat']].groupby('year').sum().loc[yy,:]
porverty_comparison_pred_wb_pf = ml_dataset_filtered_validation_world_bank.loc[:,['year','income_hat_pf']].groupby('year').sum().loc[yy,:]
porverty_comparison_pred_wb_lf = ml_dataset_filtered_validation_world_bank.loc[:,['year','income_hat_lf']].groupby('year').sum().loc[yy,:]
porverty_comparison_diff = pd.DataFrame(porverty_comparison_test.copy())

# Get difference between the true and predicted national rate:
porverty_comparison_diff.loc[:,'Forecast Oct. t-1'] = np.array(porverty_comparison_test) - np.array(porverty_comparison_pred_wb_lf)
porverty_comparison_diff.loc[:,'Gradient Boosting'] = np.array(porverty_comparison_test) - np.array(porverty_comparison_pred)
porverty_comparison_diff.loc[:,'Forecast Apr. t (WB)'] = np.array(porverty_comparison_test) - np.array(porverty_comparison_pred_wb)
porverty_comparison_diff.loc[:,'Perfect Forecast'] = np.array(porverty_comparison_test) - np.array(porverty_comparison_pred_wb_pf)

porverty_comparison_diff = porverty_comparison_diff.drop(columns=[yy])

# Plotting
plt.clf()
ax = np.abs(porverty_comparison_diff).plot.bar(figsize=(10, 6), width=0.8)
ax.set_xlabel('Average Income')
ax.set_ylabel('Difference: True - Predicted')
ax.set_xticklabels(porverty_comparison_diff.index, rotation=45)
plt.legend(title='Year',  loc='upper right')
plt.ylim([-.05, 100])
plt.tight_layout()
plt.grid(True)
# plt.show()
plt.savefig('../figures/fig3_income_pc_prediction_' + str(yy) + '.pdf', bbox_inches='tight')

print('Figure 3 saved')

# Year
yy = 2021

# Get difference between the true and predicted national rate:
porverty_comparison_test = ml_dataset_filtered_true.loc[:,['year','income_hat']].groupby('year').sum().loc[yy,:]
porverty_comparison_pred = ml_dataset_filtered_validation.loc[:,['year','income_hat']].groupby('year').sum().loc[yy,:]
porverty_comparison_pred_wb = ml_dataset_filtered_validation_world_bank.loc[:,['year','income_hat']].groupby('year').sum().loc[yy,:]
porverty_comparison_pred_wb_pf = ml_dataset_filtered_validation_world_bank.loc[:,['year','income_hat_pf']].groupby('year').sum().loc[yy,:]
porverty_comparison_pred_wb_lf = ml_dataset_filtered_validation_world_bank.loc[:,['year','income_hat_lf']].groupby('year').sum().loc[yy,:]
porverty_comparison_diff = pd.DataFrame(porverty_comparison_test.copy())

# Get difference between the true and predicted national rate:
porverty_comparison_diff.loc[:,'Forecast Oct. t-1'] = np.array(porverty_comparison_test) - np.array(porverty_comparison_pred_wb_lf)
porverty_comparison_diff.loc[:,'Gradient Boosting'] = np.array(porverty_comparison_test) - np.array(porverty_comparison_pred)
porverty_comparison_diff.loc[:,'Forecast Apr. t (WB)'] = np.array(porverty_comparison_test) - np.array(porverty_comparison_pred_wb)
porverty_comparison_diff.loc[:,'Perfect Forecast'] = np.array(porverty_comparison_test) - np.array(porverty_comparison_pred_wb_pf)

porverty_comparison_diff = porverty_comparison_diff.drop(columns=[yy])

# Plotting
plt.clf()
ax = np.abs(porverty_comparison_diff).plot.bar(figsize=(10, 6), width=0.8)
ax.set_xlabel('Average Income')
ax.set_ylabel('Difference: True - Predicted')
# ax.set_xticklabels(porverty_comparison_diff.index, rotation=45)
plt.legend(title='Year',  loc='upper right')
plt.ylim([-.05, 100])
plt.tight_layout()
plt.grid(True)
# plt.show()
plt.savefig('../figures/fig3_income_pc_prediction_' + str(yy) + '.pdf', bbox_inches='tight')

print('Figure 3 saved')



#%% Figure 3 (fig3_prediction_vs_true_poverty_rate_national): 
# Poverty Rate National (Forecast Apr. t version)
#-----------------------------------------------------------------------------------

# True data: (2017-2019)
ml_dataset_filtered_true['n_people'] = ml_dataset_filtered_true['mieperho'] * ml_dataset_filtered_true['pondera_i']
household_weight_test = ml_dataset_filtered_true['n_people']/ml_dataset_filtered_true.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_true['income_pc_weighted'] = (ml_dataset_filtered_true['income_pc'] ) * household_weight_test

# Predicted data: (using 2016 data)
ml_dataset_filtered_validation['n_people'] = ml_dataset_filtered_validation['mieperho'] * ml_dataset_filtered_validation['pondera_i']
household_weight_prediction = ml_dataset_filtered_validation['n_people']/ml_dataset_filtered_validation.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_validation['income_pc_hat_weighted'] = (ml_dataset_filtered_validation['income_pc_hat'] ) * household_weight_prediction

# Predicted data Forecast Apr. t: (using 2016 data)
ml_dataset_filtered_validation_world_bank['n_people'] = ml_dataset_filtered_validation_world_bank['mieperho'] * ml_dataset_filtered_validation_world_bank['pondera_i']
household_weight_prediction = ml_dataset_filtered_validation_world_bank['n_people']/ml_dataset_filtered_validation_world_bank.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_validation_world_bank['income_pc_hat_weighted'] = (ml_dataset_filtered_validation_world_bank['income_pc_hat'] ) * household_weight_prediction

# Predicted data Perfect Forecast: (using 2016 data)
ml_dataset_filtered_validation_world_bank['n_people'] = ml_dataset_filtered_validation_world_bank['mieperho'] * ml_dataset_filtered_validation_world_bank['pondera_i']
household_weight_prediction = ml_dataset_filtered_validation_world_bank['n_people']/ml_dataset_filtered_validation_world_bank.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_validation_world_bank['income_pc_hat_perfect_forecast_weighted'] = (ml_dataset_filtered_validation_world_bank['income_pc_hat_perfect_forecast'] ) * household_weight_prediction

# Predicted data Forecast Oct. t-1: (using 2016 data)
ml_dataset_filtered_validation_world_bank['n_people'] = ml_dataset_filtered_validation_world_bank['mieperho'] * ml_dataset_filtered_validation_world_bank['pondera_i']
household_weight_prediction = ml_dataset_filtered_validation_world_bank['n_people']/ml_dataset_filtered_validation_world_bank.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_validation_world_bank['income_pc_hat_lagged_forecast_weighted'] = (ml_dataset_filtered_validation_world_bank['income_pc_hat_lagged_forecast'] ) * household_weight_prediction


# Get difference between the true and predicted national rate:
porverty_comparison_test = ml_dataset_filtered_true.loc[:,['year','income_pc_weighted']].groupby('year').sum()
porverty_comparison_pred = ml_dataset_filtered_validation.loc[:,['year','income_pc_hat_weighted']].groupby('year').sum()
porverty_comparison_world_bank_pred = ml_dataset_filtered_validation_world_bank.loc[:,['year','income_pc_hat_weighted']].groupby('year').sum()
porverty_comparison_world_bank_pred_pf = ml_dataset_filtered_validation_world_bank.loc[:,['year','income_pc_hat_perfect_forecast_weighted']].groupby('year').sum()
porverty_comparison_world_bank_pred_lf = ml_dataset_filtered_validation_world_bank.loc[:,['year','income_pc_hat_lagged_forecast_weighted']].groupby('year').sum()

porverty_comparison_diff = porverty_comparison_test.copy()
porverty_comparison_diff['Forecast Oct. t-1'] = np.array(porverty_comparison_test) - np.array(porverty_comparison_world_bank_pred_lf)
porverty_comparison_diff['Gradient Boosting'] = np.array(porverty_comparison_test) - np.array(porverty_comparison_pred)
porverty_comparison_diff['Forecast Apr. t (WB)'] = np.array(porverty_comparison_test) - np.array(porverty_comparison_world_bank_pred)
porverty_comparison_diff['Perfect Forecast'] = np.array(porverty_comparison_test) - np.array(porverty_comparison_world_bank_pred_pf)


# Plotting
plt.clf()
ax = np.abs(porverty_comparison_diff.iloc[:,1:]).plot.bar(figsize=(10, 6), width=0.8)
ax.set_xlabel('abs(E[Predicted Income] - E[True Income])')
ax.set_ylabel('Difference: True - Predicted')
ax.set_title('Income Comparison by Year')
ax.set_xticklabels(porverty_comparison_diff.index, rotation=45)
plt.legend(title='Year',  loc='upper right')
# plt.ylim([-.05, .2])
plt.tight_layout()
plt.grid(True)
# plt.show()
plt.savefig('../figures/fig3_prediction_vs_true_average_income_national.pdf', bbox_inches='tight')

print('Figure 3 saved')


#%% Figure 4.1 (fig4_prediction_vs_true_poverty_rate_regions): 
# Replicate poverty rate (by region)
#----------------------------------------------------

grouping_variables = ['year', 'ubigeo_region']

# Get predicted poverty rate by year and region:
porverty_comparison_region = postEstimation.group_porverty_rate_for_time_series(grouping_variables,  df_true)
porverty_comparison_region = porverty_comparison_region.loc[:,['year','ubigeo_region','poor_685','poor_365','poor_215', 'poor_hat_685','poor_hat_365','poor_hat_215']].set_index(['ubigeo_region','year']).sort_index()

# Get predicted poverty rate by year and region:
porverty_comparison_region_pred = postEstimation.group_porverty_rate_for_time_series(grouping_variables,  df)
porverty_comparison_region_pred = porverty_comparison_region_pred.loc[:,['year','ubigeo_region','poor_685','poor_365','poor_215', 'poor_hat_685','poor_hat_365','poor_hat_215']].set_index(['ubigeo_region','year']).sort_index()

# Get predicted and true poverty rate by year and region according to WB:
porverty_comparison_region_wb = postEstimation.group_porverty_rate_for_time_series(grouping_variables,  df_wb)
porverty_comparison_region_wb = porverty_comparison_region_wb.loc[:,['year','ubigeo_region','poor_685','poor_365','poor_215', 'poor_hat_685','poor_hat_365','poor_hat_215']].set_index(['ubigeo_region','year']).sort_index()

# Get difference between the true and predicted national rate:
# Predicted GB 
porverty_comparison_region_diff = (porverty_comparison_region
                                    .reset_index()
                                    .loc[:,['year','ubigeo_region','poor_685','poor_365','poor_215']]
                                    .set_index(['ubigeo_region','year'])
                                    .copy()
                                    )


porverty_comparison_region_diff[['poor_685', 'poor_365',  'poor_215']] = np.array(porverty_comparison_region_pred[['poor_hat_685', 'poor_hat_365',  'poor_hat_215']]) - np.array(porverty_comparison_region[['poor_685','poor_365','poor_215']])
porverty_comparison_region_diff[['poor_wb_685', 'poor_wb_365',  'poor_wb_215']] = np.array(porverty_comparison_region_wb[['poor_hat_685', 'poor_hat_365',  'poor_hat_215']]) - np.array(porverty_comparison_region[['poor_685','poor_hat_365','poor_215']])


# Barplot with difference:

for yy in [2017, 2018, 2019, 2020, 2021]:

  plt.clf()

  n_rows = 5
  n_cols = 5
  fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))

  # Loop over each ubigeo_region
  for i, (region, data) in enumerate(porverty_comparison_region_diff.query('year == ' + str(yy)).iterrows()):

      data = data.reset_index()
      data.columns = ['index', 'PovertyRate']
      data[['Poverty Line', 'Type']] = data['index'].str.rsplit('_', n=1, expand=True)
      data = data.pivot(index='Poverty Line', columns='Type', values='PovertyRate')
      #s
      ax = axes[i // n_cols, i % n_cols]
      data.plot(kind='bar', ax=ax)
      ax.set_title(region)
      ax.set_ylabel('Rate')
      ax.set_xlabel('Poverty Line')
      ax.set_ylim(-.8, .8)
      # plt.xticks(rotation=90)

  # Hide unused subplots
  for j in range(i + 1, n_rows * n_cols):
      axes[j // n_cols, j % n_cols].axis('off')

  plt.savefig('../figures/fig4b_prediction_vs_true_poverty_rate_regions_diff_' + str(yy) + '.pdf', bbox_inches='tight')

print('Figure 4 saved')



#%% Figure 4.2 (fig4_2_prediction_vs_true_poverty_rate_regions_scatter):

# Scatter plot of predicted poverty rate vs true poverty rate
#--------------------------------------------------------------

# Gradient boosting:
#--------------------------------------------------------------

for pp in ['685', '365', '215']:

  plt.clf()
  plt.figure(figsize=(10, 10))
  sns.scatterplot(x=porverty_comparison_region.query('year == ' + str(base_year+1))['poor_' + pp], 
                    y=porverty_comparison_region_pred.query('year == ' + str(base_year+1))['poor_hat_' + pp], 
                    label=str(base_year+1), 
                    color=settings.color1,
                    s=150
                  )
  sns.scatterplot(x=porverty_comparison_region.query('year == ' + str(base_year+2))['poor_' + pp], 
                    y=porverty_comparison_region_pred.query('year == ' + str(base_year+2))['poor_hat_' + pp], 
                    label=str(base_year+2), 
                    color=settings.color2,
                    s=150
                  )
  sns.scatterplot(x=porverty_comparison_region.query('year == ' + str(base_year+3))['poor_' + pp], 
                    y=porverty_comparison_region_pred.query('year == ' + str(base_year+3))['poor_hat_' + pp], 
                    label= str(base_year+3), 
                    color=settings.color3,
                    s=150
                  )
  
  max_x_pred = (porverty_comparison_region_pred.reset_index()
                                          .query('year >= ' + str(base_year+1))
                                          .query('year <= ' + str(base_year+3))
                                          .loc[:, ['poor_' + pp, 'poor_hat_' + pp]]
                                          .max()
                                          .max())
  max_x_wb = (porverty_comparison_region.reset_index()
                                          .query('year >= ' + str(base_year+1))
                                          .query('year <= ' + str(base_year+3))
                                          .loc[:, ['poor_' + pp, 'poor_hat_' + pp]]
                                          .max()
                                          .max())
  
  max_x = max(max_x_pred, max_x_wb)


  sns.lineplot(x=[0,max_x], y=[0,max_x], color=settings.color4)
  plt.xlabel('True Poverty Rate')
  plt.ylabel('Predicted Poverty Rate')
  plt.legend()
  plt.grid(True)
  plt.savefig('../figures/fig4_2_prediction_vs_true_poverty_rate_regions_p'+ pp +  '_scatter.pdf', bbox_inches='tight')


# Forecast Apr. t:
#--------------------------------------------------------------


for pp in ['685', '365', '215']:

  plt.clf()
  plt.figure(figsize=(10, 10))
  sns.scatterplot(x=porverty_comparison_region.query('year == ' + str(base_year+1))['poor_' + pp], 
                    y=porverty_comparison_region_wb.query('year == ' + str(base_year+1))['poor_hat_' + pp], 
                    label=str(base_year+1), 
                    color=settings.color1,
                    s=150
                  )
  sns.scatterplot(x=porverty_comparison_region.query('year == ' + str(base_year+2))['poor_' + pp], 
                    y=porverty_comparison_region_wb.query('year == ' + str(base_year+2))['poor_hat_' + pp], 
                    label=str(base_year+2), 
                    color=settings.color2,
                    s=150
                  )
  sns.scatterplot(x=porverty_comparison_region.query('year == ' + str(base_year+3))['poor_' + pp], 
                    y=porverty_comparison_region_wb.query('year == ' + str(base_year+3))['poor_hat_' + pp], 
                    label=str(base_year+3), 
                    color=settings.color3,
                    s=150
                  )
  
  max_x_pred = (porverty_comparison_region_wb.reset_index()
                                          .query('year >= ' + str(base_year+1))
                                          .query('year <= ' + str(base_year+3))
                                          .loc[:, ['poor_' + pp, 'poor_hat_' + pp]]
                                          .max()
                                          .max())
  max_x_wb = (porverty_comparison_region.reset_index()
                                          .query('year >= ' + str(base_year+1))
                                          .query('year <= ' + str(base_year+3))
                                          .loc[:, ['poor_' + pp, 'poor_hat_' + pp]]
                                          .max()
                                          .max())
  
  max_x = max(max_x_pred, max_x_wb)


  sns.lineplot(x=[0,max_x], y=[0,max_x], color=settings.color4)
  plt.xlabel('True Poverty Rate')
  plt.ylabel('Predicted Poverty Rate')
  plt.legend()
  plt.grid(True)
  plt.savefig('../figures/fig4_2_prediction_wb_vs_true_poverty_rate_regions_p'+ pp +  '_scatter.pdf', bbox_inches='tight')





#%% Figure 4.1 (fig4_prediction_vs_true_poverty_rate_provincia): 
# Replicate poverty rate (by provincia])
#----------------------------------------------------


grouping_variables = ['year', 'ubigeo_provincia']

# Get predicted poverty rate by year and provincia:
porverty_comparison_provincia = postEstimation.group_porverty_rate_for_time_series(grouping_variables,  df_true)
porverty_comparison_provincia = porverty_comparison_provincia.loc[:,['year','ubigeo_provincia','poor_685','poor_365','poor_215']].set_index(grouping_variables).sort_index()

# Get predicted poverty rate by year and provincia:
porverty_comparison_provincia_pred = postEstimation.group_porverty_rate_for_time_series(grouping_variables,  df)
porverty_comparison_provincia_pred = porverty_comparison_provincia_pred.loc[:,['year','ubigeo_provincia','poor_hat_685','poor_hat_365','poor_hat_215']].set_index(grouping_variables).sort_index()

# Get predicted and true poverty rate by year and provincia according to WB:
porverty_comparison_provincia_wb = postEstimation.group_porverty_rate_for_time_series(grouping_variables,  df_wb)
porverty_comparison_provincia_wb = porverty_comparison_provincia_wb.loc[:,['year','ubigeo_provincia','poor_hat_685','poor_hat_365','poor_hat_215']].set_index(grouping_variables).sort_index()


# Gradient boosting:
#--------------------------------------------------------------

for pp in ['685', '365', '215']:

  plt.clf()
  plt.figure(figsize=(10, 10))
  sns.scatterplot(x=porverty_comparison_provincia.query('year >= ' + str(base_year+1))['poor_' + pp], 
                    y=porverty_comparison_provincia_pred.query('year >= ' + str(base_year+1))['poor_hat_' + pp], 
                    label=str(base_year+1), 
                    color=settings.color1,
                    s=150
                  )
  sns.scatterplot(x=porverty_comparison_provincia.query('year >= ' + str(base_year+2))['poor_' + pp], 
                    y=porverty_comparison_provincia_pred.query('year >= ' + str(base_year+2))['poor_hat_' + pp], 
                    label=str(base_year+2), 
                    color=settings.color2,
                    s=150
                  )
  sns.scatterplot(x=porverty_comparison_provincia.query('year >= ' + str(base_year+3))['poor_' + pp], 
                    y=porverty_comparison_provincia_pred.query('year >= ' + str(base_year+3))['poor_hat_' + pp], 
                    label=str(base_year+3), 
                    color=settings.color3,
                    s=150
                  )
  
  max_x_pred = (porverty_comparison_provincia_pred.reset_index()
                                          .query('year >=' + str(base_year+1))
                                          .query('year <=' + str(base_year+3))
                                          .loc[:, ['poor_' + pp, 'poor_hat_' + pp]]
                                          .max()
                                          .max())
  max_x_wb = (porverty_comparison_provincia.reset_index()
                                          .query('year >=' + str(base_year+1))
                                          .query('year <=' + str(base_year+3))
                                          .loc[:, ['poor_' + pp, 'poor_hat_' + pp]]
                                          .max()
                                          .max())
  
  max_x = max(max_x_pred, max_x_wb)


  sns.lineplot(x=[0,max_x], y=[0,max_x], color=settings.color4)
  plt.xlabel('True Poverty Rate')
  plt.ylabel('Predicted Poverty Rate')
  plt.legend()
  plt.grid(True)
  plt.savefig('../figures/fig4_2_prediction_vs_true_poverty_rate_provincia_p'+ pp +  '_scatter.pdf', bbox_inches='tight')


# Forecast Apr. t:
#--------------------------------------------------------------

for pp in ['685', '365', '215']:

  plt.clf()
  plt.figure(figsize=(10, 10))
  sns.scatterplot(x=porverty_comparison_provincia.query('year >= ' + str(base_year+1))['poor_' + pp], 
                    y=porverty_comparison_provincia_wb.query('year >= ' + str(base_year+1))['poor_hat_' + pp], 
                    label=str(base_year+1), 
                    color=settings.color1,
                    s=150
                  )
  sns.scatterplot(x=porverty_comparison_provincia.query('year >= ' + str(base_year+2))['poor_' + pp], 
                    y=porverty_comparison_provincia_wb.query('year >= ' + str(base_year+2))['poor_hat_' + pp], 
                    label=str(base_year+2), 
                    color=settings.color2,
                    s=150
                  )
  sns.scatterplot(x=porverty_comparison_provincia.query('year >= ' + str(base_year+3))['poor_' + pp], 
                    y=porverty_comparison_provincia_wb.query('year >= ' + str(base_year+3))['poor_hat_' + pp], 
                    label=str(base_year+3), 
                    color=settings.color3,
                    s=150
                  )
  
  max_x_pred = (porverty_comparison_provincia_wb.reset_index()
                                          .query('year >=' + str(base_year+1))
                                          .query('year <=' + str(base_year+3))
                                          .loc[:, ['poor_' + pp, 'poor_hat_' + pp]]
                                          .max()
                                          .max())
  max_x_wb = (porverty_comparison_provincia.reset_index()
                                          .query('year >=' + str(base_year+1))
                                          .query('year <=' + str(base_year+3))
                                          .loc[:, ['poor_' + pp, 'poor_hat_' + pp]]
                                          .max()
                                          .max())
  
  max_x = max(max_x_pred, max_x_wb)


  sns.lineplot(x=[0,max_x], y=[0,max_x], color=settings.color4)
  plt.xlabel('True Poverty Rate')
  plt.ylabel('Predicted Poverty Rate')
  plt.legend()
  plt.grid(True)
  plt.savefig('../figures/fig4_2_prediction_wb_vs_true_poverty_rate_provincia_p'+ pp +  '_scatter.pdf', bbox_inches='tight')


#%% Histogram or density of difference at provincial level:
  
# Get difference between the true and predicted national rate:
# Predicted GB 
porverty_comparison_provincia_diff = (porverty_comparison_provincia
                                      .merge(porverty_comparison_provincia_pred, on=['year','ubigeo_provincia'], how='left')
                                      .merge(porverty_comparison_provincia_wb, on=['year','ubigeo_provincia'], how='left')
                                      .reset_index()
                                      .set_index(['ubigeo_provincia','year'])
                                      .copy()
                                    )

porverty_comparison_provincia_diff[['diff_685', 'diff_365',  'diff_215']] = np.array(porverty_comparison_provincia_diff[['poor_hat_685_x', 'poor_hat_365_x',  'poor_hat_215_x']]) - np.array(porverty_comparison_provincia_diff[['poor_685','poor_365','poor_215']])
porverty_comparison_provincia_diff[['diff_wb_685', 'diff_wb_365',  'diff_wb_215']] = np.array(porverty_comparison_provincia_diff[['poor_hat_685_y', 'poor_hat_365_y',  'poor_hat_215_y']]) - np.array(porverty_comparison_provincia_diff[['poor_685','poor_365','poor_215']])

porverty_comparison_provincia_diff = porverty_comparison_provincia_diff.query('year >=' + str(base_year+1))


# Plotting difference at provincial level p685:
plt.clf()
plt.figure(figsize=(10, 10))
sns.histplot(porverty_comparison_provincia_diff['diff_685'], 
                color=settings.color1, 
                # kde=True, 
                label='Gradient Boosting', 
                stat='density', 
                fill=False, 
                element='step',
                bins=50
                )
sns.histplot(porverty_comparison_provincia_diff['diff_wb_685'], 
                color=settings.color2, 
                # kde=True, 
                label='World Bank', 
                stat='density', 
                fill=False, 
                element='step',
                bins=50
                )
plt.legend()
plt.xlim(-.6, .6)
plt.savefig('../figures/fig5a_prediction_vs_wb_diff_distribution_685.pdf', bbox_inches='tight')

# Plotting difference at provincial level p365:
plt.clf()
plt.figure(figsize=(10, 10))
sns.histplot(porverty_comparison_provincia_diff['diff_365'], 
                color=settings.color1, 
                # kde=True, 
                label='Gradient Boosting', 
                stat='density', 
                fill=False, 
                element='step',
                bins=50
                )
sns.histplot(porverty_comparison_provincia_diff['diff_wb_365'], 
                color=settings.color2, 
                # kde=True, 
                label='World Bank', 
                stat='density', 
                fill=False, 
                element='step',
                bins=50
                )
plt.legend()
plt.xlim(-.4, .4)
plt.savefig('../figures/fig5b_prediction_vs_wb_diff_distribution_365.pdf', bbox_inches='tight')



# Plotting difference at provincial level p215:
plt.clf()
plt.figure(figsize=(10, 10))
sns.histplot(porverty_comparison_provincia_diff['diff_215'], 
                color=settings.color1, 
                # kde=True, 
                label='Gradient Boosting', 
                stat='density', 
                fill=False, 
                element='step',
                bins=50
                )
sns.histplot(porverty_comparison_provincia_diff['diff_wb_215'], 
                color=settings.color2, 
                # kde=True, 
                label='World Bank', 
                stat='density', 
                fill=False, 
                element='step',
                bins=50
                )
plt.legend()
plt.xlim(-.3, .3)
plt.savefig('../figures/fig5c_prediction_vs_wb_diff_distribution_215.pdf', bbox_inches='tight')



#%% Figure 9 (fig9_gini_time_series): 
# Gini Coefficient
#------------------------------------------

# True data: (2017-2019)
df_true['n_people'] = df_true['mieperho'] * df_true['pondera_i']
household_weight = df_true['n_people']/df_true.groupby('year')['n_people'].transform('sum')
df_true['income_pc_weighted'] = df_true['income_pc'] # * household_weight

# Predicted data: (using 2016 data)
df['n_people'] = df['mieperho'] * df['pondera_i']
household_weight = df['n_people']/df.groupby('year')['n_people'].transform('sum')
df['income_pc_hat_weighted'] = df['income_pc_hat']  #* household_weight

# Predicted data WB: (using 2016 data)
df_wb['n_people'] = df_wb['mieperho'] * df_wb['pondera_i']
household_weight = df_wb['n_people']/df_wb.groupby('year')['n_people'].transform('sum')
df_wb['income_pc_hat_weighted'] = df_wb['income_pc_hat']  #* household_weight

def gini_coefficient(income_data):
    # Ensure the income data is sorted.
    sorted_income = np.sort(income_data)
    n = len(income_data)
    
    # Calculate the cumulative sum of the sorted incomes
    cumulative_income = np.cumsum(sorted_income)
    
    # Calculate the Gini coefficient using the simplified formula
    index_times_income = np.sum((np.arange(1, n+1) * sorted_income))
    total_income = cumulative_income[-1]  # This is the same as np.sum(sorted_income)
    gini = (2 * index_times_income) / (n * total_income) - (n + 1) / n
    
    return gini


income_series = df_true.query('income_pc < 2200').groupby('year')['income_pc_weighted'].apply(gini_coefficient).reset_index().rename(columns={'income_pc_weighted':'gini'})
income_series['gini_hat'] = list(df.query('income_pc_hat < 2200').groupby('year')['income_pc_hat_weighted'].apply(gini_coefficient))
income_series['gini_hat_wb'] = list(df_wb.query('income_pc_hat < 2200').groupby('year')['income_pc_hat_weighted'].apply(gini_coefficient))
# Convert 'year' and 'month' to a datetime
income_series['date'] = pd.to_datetime(income_series[['year']].assign(MONTH=1,DAY=1))

# Plotting:
plt.clf()
# Plotting the means with standard deviation
plt.figure(figsize=(10, 10))
plt.plot(income_series['date'], income_series['gini'], label='True GINI', color=settings.color1)
plt.plot(income_series['date'], income_series['gini_hat'], label='Predicted GINI', color=settings.color2, linestyle='--')
plt.plot(income_series['date'], income_series['gini_hat_wb'], label='Predicted GINI (WB)', color=settings.color3, linestyle='--')
plt.xlabel('Date')
plt.ylabel('Income')
# plt.ylim(0.5, .8)
plt.legend()
plt.grid(True)
# plt.show()

plt.savefig('../figures/fig9_gini_time_series.pdf', bbox_inches='tight')
print('Figure 9 saved')
print('End of code: 04_generate_prediction_report.py')
