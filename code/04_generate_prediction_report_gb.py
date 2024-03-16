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

date = '2024-03-15' #datetime.today().strftime('%Y-%m-%d')

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

year_end = 2021

ml_dataset_filtered_train = dpml.filter_ml_dataset(ml_dataset, year_end=year_end).query('year<=2016')

# Validation dataset:
ml_dataset_filtered_validation = (
                                    dpml.filter_ml_dataset(ml_dataset, year_end=year_end)
                                        .query('year >= 2017')
                                        .query('year <= ' + str(year_end))
                                        .query('true_year==2016') # Keep only observations that correspond to 2016 data
                                    )
# Validation dataset (World Bank version):
ml_dataset_filtered_validation_world_bank = (
                                    dpml.filter_ml_dataset(ml_dataset, year_end=year_end)
                                        .query('year >= 2017')
                                        .query('year <= ' + str(year_end))
                                        .query('true_year==2016') # Keep only observations that correspond to 2016 data
                                    )
# True dataset:
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


#%% Plot with standard deviation:

plt.clf()
plt.plot(df.groupby('year').predicted_error.std())
plt.ylim([0,1])
plt.savefig('../figures/std_trend.pdf', bbox_inches='tight')

#%% Figure 0 (Binned scatterplot:)
#----------------------------------------------------------------

# Run scatter plot with distribution for prediction year (default = 2016)
df_pred_year = ml_dataset_filtered_train.query('year >= 2013').query('year <= 2016').copy()

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
                  #   kde=True, 
                    label='Predicted Income', 
                    stat='density', 
                    fill=False, 
                    element='step',
                    ax=ax)
    sns.histplot(region_data_true['income_pc'], 
                    color=settings.color2, 
                  #   kde=True, 
                    label='True Income', 
                    stat='density', 
                    fill=False, 
                    element='step',
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
ax = porverty_comparison_diff.plot.bar(figsize=(10, 6), width=0.8)
ax.set_xlabel('Poverty Threshold')
ax.set_ylabel('Difference: True - Predicted')
ax.set_title('Poverty Comparison by Year')
ax.set_xticklabels(porverty_comparison_diff.index, rotation=45)
plt.legend(title='Year',  loc='upper right')
plt.ylim([-.8, .8])
plt.tight_layout()
plt.savefig('../figures/fig3_prediction_vs_true_poverty_rate_national.pdf', bbox_inches='tight')

print('Figure 3 saved')

#%% Figure 4.1 (fig4_prediction_vs_true_poverty_rate_regions): 
# Replicate poverty rate (by region)
#----------------------------------------------------

# True data: (2017-2019)
ml_dataset_filtered_true['n_people'] = ml_dataset_filtered_true['mieperho'] * ml_dataset_filtered_true['pondera_i']
household_weight_test = ml_dataset_filtered_true['n_people']/ml_dataset_filtered_true.groupby(['year', 'ubigeo_region'])['n_people'].transform('sum')
ml_dataset_filtered_true['poor_685'] = (ml_dataset_filtered_true['income_pc'] <= ml_dataset_filtered_true['lp_685usd_ppp']) * household_weight_test
ml_dataset_filtered_true['poor_365'] = (ml_dataset_filtered_true['income_pc'] <= ml_dataset_filtered_true['lp_365usd_ppp']) * household_weight_test
ml_dataset_filtered_true['poor_215'] = (ml_dataset_filtered_true['income_pc'] <= ml_dataset_filtered_true['lp_215usd_ppp']) * household_weight_test

# Predicted data: (using 2016 data)
ml_dataset_filtered_validation['n_people'] = ml_dataset_filtered_validation['mieperho'] * ml_dataset_filtered_validation['pondera_i']
household_weight_prediction = ml_dataset_filtered_validation['n_people']/ml_dataset_filtered_validation.groupby(['year', 'ubigeo_region'])['n_people'].transform('sum')
ml_dataset_filtered_validation['poor_hat_685'] = (ml_dataset_filtered_validation['income_pc_hat'] <= ml_dataset_filtered_validation['lp_685usd_ppp']) * household_weight_prediction
ml_dataset_filtered_validation['poor_hat_365'] = (ml_dataset_filtered_validation['income_pc_hat'] <= ml_dataset_filtered_validation['lp_365usd_ppp']) * household_weight_prediction
ml_dataset_filtered_validation['poor_hat_215'] = (ml_dataset_filtered_validation['income_pc_hat'] <= ml_dataset_filtered_validation['lp_215usd_ppp']) * household_weight_prediction

# Get predicted and true poverty rate by year and region:
porverty_comparison_test = ml_dataset_filtered_true.loc[:,['year','ubigeo_region','poor_685','poor_365','poor_215']].groupby(['ubigeo_region', 'year']).sum()
porverty_comparison_pred = ml_dataset_filtered_validation.loc[:,['year','ubigeo_region','poor_hat_685','poor_hat_365','poor_hat_215']].groupby(['ubigeo_region', 'year']).sum()
porverty_comparison_region = pd.concat([porverty_comparison_test, porverty_comparison_pred], axis=1)


plt.clf()

n_rows = 5
n_cols = 5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))

# Loop over each ubigeo_region
for i, (region, data) in enumerate(porverty_comparison_region.query('year == 2017').iterrows()):
    #
    data = data.reset_index()
    data.columns = ['index', 'PovertyRate']
    data[['Poverty Line', 'Type']] = data['index'].str.rsplit('_', n=1, expand=True)
    data = data.pivot(index='Poverty Line', columns='Type', values='PovertyRate')
    #
    ax = axes[i // n_cols, i % n_cols]
    data.plot(kind='bar', ax=ax)
    ax.set_title(region)
    ax.set_ylabel('Rate')
    ax.set_xlabel('Poverty Line')
    ax.set_ylim(0, .8)
    plt.xticks(rotation=90)

# Hide unused subplots
for j in range(i + 1, n_rows * n_cols):
    axes[j // n_cols, j % n_cols].axis('off')

plt.savefig('../figures/fig4_prediction_vs_true_poverty_rate_regions.pdf', bbox_inches='tight')

print('Figure 4 saved')




#%% Figure 4.2 (fig4_2_prediction_vs_true_poverty_rate_regions_scatter):

# Scatter plot of predicted poverty rate vs true poverty rate
#--------------------------------------------------------------

# Poverty 685:
#--------------------------------------------------------------

plt.clf()
plt.figure(figsize=(10, 10))
sns.scatterplot(x=porverty_comparison_region.query('year == 2017')['poor_685'], 
                  y=porverty_comparison_region.query('year == 2017')['poor_hat_685'], 
                  label='2017', 
                  color=settings.color1,
                  s=150
                )
sns.scatterplot(x=porverty_comparison_region.query('year == 2018')['poor_685'], 
                  y=porverty_comparison_region.query('year == 2018')['poor_hat_685'], 
                  label='2018', 
                  color=settings.color2,
                  s=150
                )
sns.scatterplot(x=porverty_comparison_region.query('year == 2019')['poor_685'], 
                  y=porverty_comparison_region.query('year == 2019')['poor_hat_685'], 
                  label='2019', 
                  color=settings.color3,
                  s=150
                )
sns.lineplot(x=[0,.7], y=[0,.7], color=settings.color4)
plt.xlabel('True Poverty Rate')
plt.ylabel('Predicted Poverty Rate')
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig4_2_prediction_vs_true_poverty_rate_regions_p685_scatter.pdf', bbox_inches='tight')



# Poverty 365:
#--------------------------------------------------------------

plt.clf()
plt.figure(figsize=(10, 10))
sns.scatterplot(x=porverty_comparison_region.query('year == 2017')['poor_365'], 
                  y=porverty_comparison_region.query('year == 2017')['poor_hat_365'], 
                  label='2017', 
                  color=settings.color1,
                  s=150
                )
sns.scatterplot(x=porverty_comparison_region.query('year == 2018')['poor_365'], 
                  y=porverty_comparison_region.query('year == 2018')['poor_hat_365'], 
                  label='2018', 
                  color=settings.color2,
                  s=150
                )
sns.scatterplot(x=porverty_comparison_region.query('year == 2019')['poor_365'], 
                  y=porverty_comparison_region.query('year == 2019')['poor_hat_365'], 
                  label='2019', 
                  color=settings.color3,
                  s=150
                )

sns.lineplot(x=[0,.31], y=[0,.31], color=settings.color2)
plt.xlabel('True Poverty Rate')
plt.ylabel('Predicted Poverty Rate')
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig4_2_prediction_vs_true_poverty_rate_regions_p365_scatter.pdf', bbox_inches='tight')


# Poverty 215:
#--------------------------------------------------------------

plt.clf()
plt.figure(figsize=(10, 10))
sns.scatterplot(x=porverty_comparison_region.query('year == 2017')['poor_215'], 
                  y=porverty_comparison_region.query('year == 2017')['poor_hat_215'], 
                  label='2017', 
                  color=settings.color1,
                  s=150
                )
sns.scatterplot(x=porverty_comparison_region.query('year == 2018')['poor_215'], 
                  y=porverty_comparison_region.query('year == 2018')['poor_hat_215'], 
                  label='2018', 
                  color=settings.color2,
                  s=150
                )
sns.scatterplot(x=porverty_comparison_region.query('year == 2019')['poor_215'], 
                  y=porverty_comparison_region.query('year == 2019')['poor_hat_215'], 
                  label='2019', 
                  color=settings.color3,
                  s=150
                )

sns.lineplot(x=[0,.125], y=[0,.125], color=settings.color2)
plt.xlabel('True Poverty Rate')
plt.ylabel('Predicted Poverty Rate')
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig4_2_prediction_vs_true_poverty_rate_regions_p215_scatter.pdf', bbox_inches='tight')


# get mse for regional comparison:
porverty_comparison_regional_mse = porverty_comparison_region.reset_index().copy()
porverty_comparison_regional_mse['sq_error_685'] = (porverty_comparison_regional_mse['poor_685']-porverty_comparison_regional_mse['poor_hat_685'])
porverty_comparison_regional_mse['sq_error_365'] = (porverty_comparison_regional_mse['poor_365']- porverty_comparison_regional_mse['poor_hat_365'])
porverty_comparison_regional_mse['sq_error_215'] = (porverty_comparison_regional_mse['poor_215']- porverty_comparison_regional_mse['poor_hat_215'])
pov_regional_mse = porverty_comparison_regional_mse.groupby('year')[['sq_error_685', 'sq_error_365', 'sq_error_215']].mean()
pov_regional_mse['type']='regional'



#%% Figure 4.1 (fig4_prediction_vs_true_poverty_rate_provincia): 
# Replicate poverty rate (by provincia])
#----------------------------------------------------

# True data: (2017-2019)
ml_dataset_filtered_true['n_people'] = ml_dataset_filtered_true['mieperho'] * ml_dataset_filtered_true['pondera_i']
household_weight_test = ml_dataset_filtered_true['n_people']/ml_dataset_filtered_true.groupby(['year', 'ubigeo_provincia'])['n_people'].transform('sum')
ml_dataset_filtered_true['poor_685'] = (ml_dataset_filtered_true['income_pc'] <= ml_dataset_filtered_true['lp_685usd_ppp']) * household_weight_test
ml_dataset_filtered_true['poor_365'] = (ml_dataset_filtered_true['income_pc'] <= ml_dataset_filtered_true['lp_365usd_ppp']) * household_weight_test
ml_dataset_filtered_true['poor_215'] = (ml_dataset_filtered_true['income_pc'] <= ml_dataset_filtered_true['lp_215usd_ppp']) * household_weight_test

# Predicted data: (using 2016 data)
ml_dataset_filtered_validation['n_people'] = ml_dataset_filtered_validation['mieperho'] * ml_dataset_filtered_validation['pondera_i']
household_weight_prediction = ml_dataset_filtered_validation['n_people']/ml_dataset_filtered_validation.groupby(['year', 'ubigeo_provincia'])['n_people'].transform('sum')
ml_dataset_filtered_validation['poor_hat_685'] = (ml_dataset_filtered_validation['income_pc_hat'] <= ml_dataset_filtered_validation['lp_685usd_ppp']) * household_weight_prediction
ml_dataset_filtered_validation['poor_hat_365'] = (ml_dataset_filtered_validation['income_pc_hat'] <= ml_dataset_filtered_validation['lp_365usd_ppp']) * household_weight_prediction
ml_dataset_filtered_validation['poor_hat_215'] = (ml_dataset_filtered_validation['income_pc_hat'] <= ml_dataset_filtered_validation['lp_215usd_ppp']) * household_weight_prediction

# Get predicted and true poverty rate by year and region:
porverty_comparison_test = ml_dataset_filtered_true.loc[:,['year','ubigeo_provincia','poor_685','poor_365','poor_215']].groupby(['ubigeo_provincia', 'year']).sum()
porverty_comparison_pred = ml_dataset_filtered_validation.loc[:,['year','ubigeo_provincia','poor_hat_685','poor_hat_365','poor_hat_215']].groupby(['ubigeo_provincia', 'year']).sum()
porverty_comparison_provincia = pd.concat([porverty_comparison_test, porverty_comparison_pred], axis=1)



# Poverty 685:
#--------------------------------------------------------------

plt.clf()
plt.figure(figsize=(10, 10))
sns.scatterplot(x=porverty_comparison_provincia.query('year == 2017')['poor_685'], 
                  y=porverty_comparison_provincia.query('year == 2017')['poor_hat_685'], 
                  label='2017', 
                  color=settings.color1,
                  s=150
                )
sns.scatterplot(x=porverty_comparison_provincia.query('year == 2018')['poor_685'], 
                  y=porverty_comparison_provincia.query('year == 2018')['poor_hat_685'], 
                  label='2018', 
                  color=settings.color2,
                  s=150
                )
sns.scatterplot(x=porverty_comparison_provincia.query('year == 2019')['poor_685'], 
                  y=porverty_comparison_provincia.query('year == 2019')['poor_hat_685'], 
                  label='2019', 
                  color=settings.color3,
                  s=150
                )
sns.lineplot(x=[0,1], y=[0,1], color=settings.color4)
plt.xlabel('True Poverty Rate')
plt.ylabel('Predicted Poverty Rate')
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig4_3_prediction_vs_true_poverty_rate_provincia_p685_scatter.pdf', bbox_inches='tight')



# Poverty 365:
#--------------------------------------------------------------

plt.clf()
plt.figure(figsize=(10, 10))
sns.scatterplot(x=porverty_comparison_provincia.query('year == 2017')['poor_365'], 
                  y=porverty_comparison_provincia.query('year == 2017')['poor_hat_365'], 
                  label='2017', 
                  color=settings.color1,
                  s=150
                )
sns.scatterplot(x=porverty_comparison_provincia.query('year == 2018')['poor_365'], 
                  y=porverty_comparison_provincia.query('year == 2018')['poor_hat_365'], 
                  label='2018', 
                  color=settings.color2,
                  s=150
                )
sns.scatterplot(x=porverty_comparison_provincia.query('year == 2019')['poor_365'], 
                  y=porverty_comparison_provincia.query('year == 2019')['poor_hat_365'], 
                  label='2019', 
                  color=settings.color3,
                  s=150
                )

sns.lineplot(x=[0,.31], y=[0,.31], color=settings.color2)
plt.xlabel('True Poverty Rate')
plt.ylabel('Predicted Poverty Rate')
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig4_2_prediction_vs_true_poverty_rate_provincia_p365_scatter.pdf', bbox_inches='tight')


# Poverty 215:
#--------------------------------------------------------------

plt.clf()
plt.figure(figsize=(10, 10))
sns.scatterplot(x=porverty_comparison_provincia.query('year == 2017')['poor_215'], 
                  y=porverty_comparison_provincia.query('year == 2017')['poor_hat_215'], 
                  label='2017', 
                  color=settings.color1,
                  s=150
                )
sns.scatterplot(x=porverty_comparison_provincia.query('year == 2018')['poor_215'], 
                  y=porverty_comparison_provincia.query('year == 2018')['poor_hat_215'], 
                  label='2018', 
                  color=settings.color2,
                  s=150
                )
sns.scatterplot(x=porverty_comparison_provincia.query('year == 2019')['poor_215'], 
                  y=porverty_comparison_provincia.query('year == 2019')['poor_hat_215'], 
                  label='2019', 
                  color=settings.color3,
                  s=150
                )

sns.lineplot(x=[0,.125], y=[0,.125], color=settings.color2)
plt.xlabel('True Poverty Rate')
plt.ylabel('Predicted Poverty Rate')
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig4_2_prediction_vs_true_poverty_rate_provincia_p215_scatter.pdf', bbox_inches='tight')


# get mse for provincial comparison:
porverty_comparison_provincial_mse = porverty_comparison_provincia.reset_index().copy()
porverty_comparison_provincial_mse['sq_error_685'] = (porverty_comparison_provincial_mse['poor_685']-porverty_comparison_provincial_mse['poor_hat_685'])
porverty_comparison_provincial_mse['sq_error_365'] = (porverty_comparison_provincial_mse['poor_365']- porverty_comparison_provincial_mse['poor_hat_365'])
porverty_comparison_provincial_mse['sq_error_215'] = (porverty_comparison_provincial_mse['poor_215']- porverty_comparison_provincial_mse['poor_hat_215'])
pov_provincial_mse = porverty_comparison_provincial_mse.groupby('year')[['sq_error_685', 'sq_error_365', 'sq_error_215']].mean()
pov_provincial_mse['type']= 'provincial'

pov_mse = pd.concat([pov_regional_mse, pov_provincial_mse], axis=0)

pov_mse.to_csv('../tables/table1_pov_mse.csv')


