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
from joblib import Parallel, delayed, dump, load
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.linear_model import Lasso
import numpy as np
from joblib import Parallel, delayed
from global_settings import global_settings



#%% Get current working directory and parameters:

# Parameters
dataPath = 'J:/My Drive/PovertyPredictionRealTime/data'

dataPath = '/home/fcalle0/datasets/WorldBankPovertyPrediction/'

freq = 'm'

date = '2024-02-23' #datetime.today().strftime('%Y-%m-%d')

settings = global_settings()

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

# 2. Obtain filtered dataset:

ml_dataset_filtered_train = dpml.filter_ml_dataset(ml_dataset).query('year<=2018')

Y_standardized_train, X_standardized_train, scaler_X_train, scaler_Y_train = dpml.get_depvar_and_features(ml_dataset_filtered_train)

ml_dataset_filtered_validation = dpml.filter_ml_dataset(ml_dataset).query('year==2019')

Y_standardized_validation, X_standardized_validation, scaler_X_validation, scaler_Y_validation = dpml.get_depvar_and_features(ml_dataset_filtered_validation, scaler_X_train, scaler_Y_train)

# 3. Load best model:

best_model_lasso = dpml.load_ml_model(model_filename = 'best_weighted_lasso_model.joblib')

best_model_gb = dpml.load_ml_model(model_filename = 'best_weighted_gb_model.joblib')

# 4. Keep variables used in Gradient Boosting model:

# Train:
X_standardized_train =  X_standardized_train[X_standardized_train.columns[best_model_lasso.coef_ !=0]]
X_standardized_train['const'] = 1
# Validation:
X_standardized_validation =  X_standardized_validation[X_standardized_validation.columns[best_model_lasso.coef_ !=0]]
X_standardized_validation['const'] = 1

# 5. Predict income
# Train:
predicted_income = best_model_gb.predict(X_standardized_train)
random_shock = np.random.normal(loc=0, scale=predicted_income.std() * 1, size=pd.Series(predicted_income).dropna().shape[0])

ml_dataset_filtered_train['log_income_pc_hat'] = predicted_income + random_shock
ml_dataset_filtered_train['income_pc_hat'] = np.exp(ml_dataset_filtered_train['log_income_pc_hat'] * scaler_Y_train.scale_[0] + scaler_Y_train.mean_[0]) 

# Validation:
predicted_income_validation = best_model_gb.predict(X_standardized_validation)
random_shock_validation = np.random.normal(loc=0, scale=predicted_income_validation.std() * 1.1, size=pd.Series(predicted_income_validation).dropna().shape[0])

ml_dataset_filtered_validation['log_income_pc_hat'] = best_model_gb.predict(X_standardized_validation) + random_shock_validation
ml_dataset_filtered_validation['income_pc_hat'] = np.exp(ml_dataset_filtered_validation['log_income_pc_hat'] * scaler_Y_train.scale_[0] + scaler_Y_train.mean_[0]) 




# 5. Compiling both datasets and creating some variables:

df = pd.concat([ml_dataset_filtered_train, ml_dataset_filtered_validation], axis=0)

# df['income_pc_hat'] = df['income_pc_hat'] * 1.2

month_to_quarter = {1:1, 2:1, 3:1, 
                    4:4, 5:4, 6:4, 
                    7:7, 8:7, 9:7, 
                    10:10, 11:10, 12:10}

df['quarter'] = df['month'].map(month_to_quarter)

df['n_people'] = 1
# df['n_people'] = df['mieperho'] * df['pondera_i']



#%% Figure 1 (fig1_prediction_vs_true_income_distribution_lasso_training_weighted): 
# Distribution of predicted income vs true income
#--------------------------------------------------------------

plt.clf()
plt.figure(figsize=(10, 10))
sns.histplot(ml_dataset_filtered_validation['income_pc_hat'], 
                color=settings.color1, kde=True, 
                label='Predicted Income', 
                stat='density', 
                fill=False, 
                element='step'
                )
sns.histplot(ml_dataset_filtered_validation['income_pc'], 
                color=settings.color2, 
                kde=True, 
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
sns.ecdfplot(ml_dataset_filtered_validation['income_pc'], color=settings.color2, label='True Income')
plt.xlim(0, 2500)
plt.legend()
plt.xlabel('Income')
plt.ylabel('Cumulative Distribution')
plt.savefig('../figures/fig1b_prediction_vs_true_income_ecdf_lasso_training_weighted.pdf', bbox_inches='tight')
print('Figure 1b saved')

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
    sns.histplot(region_data['income_pc_hat'], 
                    color=settings.color1, 
                    kde=True, 
                    label='Predicted Income', 
                    stat='density', 
                    fill=False, 
                    element='step',
                    ax=ax)
    sns.histplot(region_data['income_pc'], 
                    color=settings.color2, 
                    kde=True, 
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

ml_dataset_filtered_validation['n_people'] = ml_dataset_filtered_validation['mieperho'] * ml_dataset_filtered_validation['pondera_i']
household_weight = ml_dataset_filtered_validation['n_people']/ml_dataset_filtered_validation.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_validation['poor_685'] = (ml_dataset_filtered_validation['income_pc'] <= ml_dataset_filtered_validation['lp_685usd_ppp']) * household_weight
ml_dataset_filtered_validation['poor_365'] = (ml_dataset_filtered_validation['income_pc'] <= ml_dataset_filtered_validation['lp_365usd_ppp']) * household_weight
ml_dataset_filtered_validation['poor_215'] = (ml_dataset_filtered_validation['income_pc'] <= ml_dataset_filtered_validation['lp_215usd_ppp']) * household_weight
ml_dataset_filtered_validation['poor_hat_685'] = (ml_dataset_filtered_validation['income_pc_hat'] <= ml_dataset_filtered_validation['lp_685usd_ppp']) * household_weight
ml_dataset_filtered_validation['poor_hat_365'] = (ml_dataset_filtered_validation['income_pc_hat'] <= ml_dataset_filtered_validation['lp_365usd_ppp']) * household_weight
ml_dataset_filtered_validation['poor_hat_215'] = (ml_dataset_filtered_validation['income_pc_hat'] <= ml_dataset_filtered_validation['lp_215usd_ppp']) * household_weight

porverty_comparison = ml_dataset_filtered_validation.loc[:,['poor_685','poor_365','poor_215', 
                                                'poor_hat_685','poor_hat_365','poor_hat_215']].sum()

porverty_comparison = pd.DataFrame(porverty_comparison).rename(columns={0:'PovertyRate'}).reset_index()

porverty_comparison[['Poverty Line', 'Type']] = porverty_comparison['index'].str.rsplit('_', n=1, expand=True)

porverty_comparison = porverty_comparison.pivot(index='Poverty Line', columns='Type', values='PovertyRate')


plt.clf()
porverty_comparison.plot(kind='bar', figsize=(10, 10), color=[settings.color1, settings.color2]) 
plt.ylabel('Sum')
plt.xlabel('Poverty Line')
plt.xticks(rotation=45)
plt.ylim(0, .5)
plt.savefig('../figures/fig3_prediction_vs_true_poverty_rate_national.pdf', bbox_inches='tight')

print('Figure 3 saved')

#%% Figure 4.1 (fig4_prediction_vs_true_poverty_rate_regions): 
# Replicate poverty rate (by region)
#----------------------------------------------------

household_weight = ml_dataset_filtered_validation['n_people']/ml_dataset_filtered_validation.groupby(['year','ubigeo_region'])['n_people'].transform('sum')
ml_dataset_filtered_validation['poor_685'] = (ml_dataset_filtered_validation['income_pc'] <= ml_dataset_filtered_validation['lp_685usd_ppp']) * household_weight
ml_dataset_filtered_validation['poor_365'] = (ml_dataset_filtered_validation['income_pc'] <= ml_dataset_filtered_validation['lp_365usd_ppp']) * household_weight
ml_dataset_filtered_validation['poor_215'] = (ml_dataset_filtered_validation['income_pc'] <= ml_dataset_filtered_validation['lp_215usd_ppp']) * household_weight
ml_dataset_filtered_validation['poor_hat_685'] = (ml_dataset_filtered_validation['income_pc_hat'] <= ml_dataset_filtered_validation['lp_685usd_ppp']) * household_weight
ml_dataset_filtered_validation['poor_hat_365'] = (ml_dataset_filtered_validation['income_pc_hat'] <= ml_dataset_filtered_validation['lp_365usd_ppp']) * household_weight
ml_dataset_filtered_validation['poor_hat_215'] = (ml_dataset_filtered_validation['income_pc_hat'] <= ml_dataset_filtered_validation['lp_215usd_ppp']) * household_weight

porverty_comparison_region = ml_dataset_filtered_validation.loc[:,['ubigeo_region','poor_685','poor_365','poor_215', 
                                                'poor_hat_685','poor_hat_365','poor_hat_215']].groupby('ubigeo_region').sum()


plt.clf()

n_rows = 5
n_cols = 5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))

# Loop over each ubigeo_region
for i, (region, data) in enumerate(porverty_comparison_region.iterrows()):
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
plt.clf()
plt.figure(figsize=(10, 10))
sns.scatterplot(x=porverty_comparison_region['poor_685'], y=porverty_comparison_region['poor_hat_685'], label='LP 685', color=settings.color1)
sns.lineplot(x=[0,.7], y=[0,.7], color=settings.color2)
plt.xlabel('True Poverty Rate')
plt.ylabel('Predicted Poverty Rate')
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig4_2_prediction_vs_true_poverty_rate_regions_p685_scatter.pdf', bbox_inches='tight')


# Poverty 365:
plt.clf()
plt.figure(figsize=(10, 10))
sns.scatterplot(x=porverty_comparison_region['poor_365'], y=porverty_comparison_region['poor_hat_365'], label='LP 365', color=settings.color1)
sns.lineplot(x=[0,.31], y=[0,.31], color=settings.color2)
plt.xlabel('True Poverty Rate')
plt.ylabel('Predicted Poverty Rate')
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig4_2_prediction_vs_true_poverty_rate_regions_p365_scatter.pdf', bbox_inches='tight')


# Poverty 215:
plt.clf()
plt.figure(figsize=(10, 10))
sns.scatterplot(x=porverty_comparison_region['poor_215'], y=porverty_comparison_region['poor_hat_215'], label='LP 215', color=settings.color1)
sns.lineplot(x=[0,.125], y=[0,.125], color=settings.color2)
plt.xlabel('True Poverty Rate')
plt.ylabel('Predicted Poverty Rate')
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig4_2_prediction_vs_true_poverty_rate_regions_p215_scatter.pdf', bbox_inches='tight')





#%% Figure 5 (fig5_average_income_time_series): 
# Time series of average income (Yearly)
#----------------------------------------------------

household_weight = df['n_people']/df.groupby('year')['n_people'].transform('sum')

df['income_pc_weighted'] = df['income_pc'] * household_weight 
df['income_pc_hat_weighted'] = df['income_pc_hat'] * household_weight 

income_series = (df.groupby(['year'])
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
income_series['date'] = pd.to_datetime(income_series[['year']].assign(MONTH=1,DAY=1))

# Plotting:
plt.clf()
plt.figure(figsize=(10, 10))

# Plotting the means with standard deviation
plt.errorbar(income_series['date'], income_series['income_pc_weighted'], yerr=income_series['std_mean'], 
             label='True Income', color=settings.color1, fmt='-')
plt.errorbar(income_series['date'], income_series['income_pc_hat_weighted'], yerr=income_series['std_hat_mean'], 
             label='Predicted Income', color=settings.color2, fmt='-.', linestyle='-.')  # Adjust linestyle if needed

plt.xlabel('Date')
plt.ylabel('Income')
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig5_average_income_time_series.pdf', bbox_inches='tight')

print('Figure 5 saved')

#%% Figure 6 (fig6_average_income_time_series_by_area): 
# Time series average income plot by area (Yearly)
#----------------------------------------------------

grouping_variables = ['year','urbano']
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
income_series['date'] = pd.to_datetime(income_series[['year']].assign(MONTH=1,DAY=1))

income_series_urban = income_series.query('urbano==1')
income_series_rural = income_series.query('urbano==0')

# Plotting the means with standard deviation
# Urbano
plt.clf()
plt.figure(figsize=(10, 10))
plt.errorbar(income_series_urban['date'], income_series_urban['income_pc_weighted'], yerr=income_series_urban['std_mean'], 
             label='True Urban', color=settings.color1, fmt='-')
plt.errorbar(income_series_urban['date'], income_series_urban['income_pc_hat_weighted'], yerr=income_series_urban['std_hat_mean'], 
             label='Predicted Urban', color=settings.color1, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
# Rural
plt.errorbar(income_series_rural['date'], income_series_rural['income_pc_weighted'], yerr=income_series_rural['std_mean'], 
             label='True Rural', color=settings.color4, fmt='-')
plt.errorbar(income_series_rural['date'], income_series_rural['income_pc_hat_weighted'], yerr=income_series_rural['std_hat_mean'], 
             label='Predicted Rural', color=settings.color4, fmt='-.', linestyle='-.')  # Adjust linestyle if needed

plt.xlabel('Date')
plt.ylabel('Income')
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig6_average_income_time_series_by_area.pdf', bbox_inches='tight')

print('Figure 6 saved')

#%% Figure 7: (fig7_average_income_time_series_quarterly)
# Time series average income (Quarterly)
#----------------------------------------------------

household_weight = df['n_people']/df.groupby(['year', 'quarter'])['n_people'].transform('sum')

df['income_pc_weighted'] = df['income_pc'] * household_weight 
df['income_pc_hat_weighted'] = df['income_pc_hat'] * household_weight 

income_series = (df.groupby(['year', 'quarter'])
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
income_series['date'] = pd.to_datetime(income_series.rename(columns={'quarter':'month'})[['year','month']].assign(DAY=1))

# Plotting:
plt.clf()
plt.figure(figsize=(10, 10))

# Plotting the means with standard deviation
plt.errorbar(income_series['date'], income_series['income_pc_weighted'], yerr=income_series['std_mean'], 
             label='True Income', color=settings.color2, fmt='-')
plt.errorbar(income_series['date'], income_series['income_pc_hat_weighted'], yerr=income_series['std_hat_mean'], 
             label='Predicted Income', color=settings.color4, fmt='--', linestyle='--')  # Adjust linestyle if needed

plt.xlabel('Date')
plt.ylabel('Income')
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig7_average_income_time_series_quarterly.pdf', bbox_inches='tight')

print('Figure 7 saved')


#%% Figure 8 (fig8_poverty_rate_time_series): 
# Poverty Rate (Yearly)
#-------------------------------------------

household_weight = df['n_people']/df.groupby('year')['n_people'].transform('sum')
df['poor_685'] = (df['income_pc'] <= df['lp_685usd_ppp']) * household_weight
df['poor_365'] = (df['income_pc'] <= df['lp_365usd_ppp']) * household_weight
df['poor_215'] = (df['income_pc'] <= df['lp_215usd_ppp']) * household_weight
df['poor_hat_685'] = (df['income_pc_hat'] <= df['lp_685usd_ppp']) * household_weight
df['poor_hat_365'] = (df['income_pc_hat'] <= df['lp_365usd_ppp']) * household_weight
df['poor_hat_215'] = (df['income_pc_hat'] <= df['lp_215usd_ppp']) * household_weight
income_series = (df.groupby(['year'])
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
income_series['date'] = pd.to_datetime(income_series[['year']].assign(MONTH=1,DAY=1))

# Plotting:
plt.clf()
plt.figure(figsize=(10, 10))
# Plotting the means with standard deviation
# poor_685
plt.errorbar(income_series['date'], income_series['poor_685'], yerr=income_series['std_685_mean'], 
             label='LP 685', color=settings.color1, fmt='-')
plt.errorbar(income_series['date'], income_series['poor_hat_685'], yerr=income_series['std_685_mean'], 
             label='LP 685 Predict', color=settings.color1, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series['date'], income_series['poor_365'], yerr=income_series['std_365_mean'], 
             label='LP 365', color=settings.color3, fmt='-')
plt.errorbar(income_series['date'], income_series['poor_hat_365'], yerr=income_series['std_365_mean'], 
             label='LP 365 Predict', color=settings.color3, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series['date'], income_series['poor_215'], yerr=income_series['std_215_mean'], 
             label='LP 215', color=settings.color5, fmt='-')
plt.errorbar(income_series['date'], income_series['poor_hat_215'], yerr=income_series['std_215_mean'], 
             label='LP 215 Predict', color=settings.color5, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.xlabel('Date')
plt.ylabel('Poverty Rate')
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig8_poverty_rate_time_series.pdf', bbox_inches='tight')

print('Figure 8 saved')


#%% Figure 8a (fig8a_poverty_rate_time_series_urbano): 
# Poverty Rate Urbano (Yearly)
#-------------------------------------------

household_weight = df['n_people']/df.groupby(['year','urbano'])['n_people'].transform('sum')
df['poor_685'] = (df['income_pc'] <= df['lp_685usd_ppp']) * household_weight
df['poor_365'] = (df['income_pc'] <= df['lp_365usd_ppp']) * household_weight
df['poor_215'] = (df['income_pc'] <= df['lp_215usd_ppp']) * household_weight
df['poor_hat_685'] = (df['income_pc_hat'] <= df['lp_685usd_ppp']) * household_weight
df['poor_hat_365'] = (df['income_pc_hat'] <= df['lp_365usd_ppp']) * household_weight
df['poor_hat_215'] = (df['income_pc_hat'] <= df['lp_215usd_ppp']) * household_weight
income_series = (df.groupby(['year','urbano'])
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
income_series['date'] = pd.to_datetime(income_series[['year']].assign(MONTH=1,DAY=1))

# Split between urbano and rural:
income_series_urban = income_series.query('urbano==1')
income_series_rural = income_series.query('urbano==0')

# Plotting:
plt.clf()
plt.figure(figsize=(10, 10))
# Plotting the means with standard deviation
# poor_685
plt.errorbar(income_series_urban['date'], income_series_urban['poor_685'], yerr=income_series_urban['std_685_mean'], 
             label='LP 685', color=settings.color1, fmt='-')
plt.errorbar(income_series_urban['date'], income_series_urban['poor_hat_685'], yerr=income_series_urban['std_685_mean'], 
             label='LP 685 Predict', color=settings.color1, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_urban['date'], income_series_urban['poor_365'], yerr=income_series_urban['std_365_mean'], 
             label='LP 365', color=settings.color3, fmt='-')
plt.errorbar(income_series_urban['date'], income_series_urban['poor_hat_365'], yerr=income_series_urban['std_365_mean'], 
             label='LP 365 Predict', color=settings.color3, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_urban['date'], income_series_urban['poor_215'], yerr=income_series_urban['std_215_mean'], 
             label='LP 215', color=settings.color5, fmt='-')
plt.errorbar(income_series_urban['date'], income_series_urban['poor_hat_215'], yerr=income_series_urban['std_215_mean'], 
             label='LP 215 Predict', color=settings.color5, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.xlabel('Date')
plt.ylabel('Poverty Rate')
plt.legend()
plt.grid(True)
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
plt.errorbar(income_series_rural['date'], income_series_rural['poor_685'], yerr=income_series_rural['std_685_mean'], 
             label='LP 685', color=settings.color1, fmt='-')
plt.errorbar(income_series_rural['date'], income_series_rural['poor_hat_685'], yerr=income_series_rural['std_685_mean'], 
             label='LP 685 Predict', color=settings.color1, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_rural['date'], income_series_rural['poor_365'], yerr=income_series_rural['std_365_mean'], 
             label='LP 365', color=settings.color3, fmt='-')
plt.errorbar(income_series_rural['date'], income_series_rural['poor_hat_365'], yerr=income_series_rural['std_365_mean'], 
             label='LP 365 Predict', color=settings.color3, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_rural['date'], income_series_rural['poor_215'], yerr=income_series_rural['std_215_mean'], 
             label='LP 215', color=settings.color5, fmt='-')
plt.errorbar(income_series_rural['date'], income_series_rural['poor_hat_215'], yerr=income_series_rural['std_215_mean'], 
             label='LP 215 Predict', color=settings.color5, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.xlabel('Date')
plt.ylabel('Poverty Rate')
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig8b_poverty_rate_time_series_rural.pdf', bbox_inches='tight')

print('Figure 8b saved')

#%% Figure 8c (fig8a_poverty_rate_time_series_urbano): 
# Poverty Rate Urbano (Yearly)
#-------------------------------------------

df_urban = df.query('urbano==1')

household_weight = df_urban['n_people']/df_urban.groupby(['year','lima_metropolitana'])['n_people'].transform('sum')
df_urban['poor_685'] = (df_urban['income_pc'] <= df_urban['lp_685usd_ppp']) * household_weight
df_urban['poor_365'] = (df_urban['income_pc'] <= df_urban['lp_365usd_ppp']) * household_weight
df_urban['poor_215'] = (df_urban['income_pc'] <= df_urban['lp_215usd_ppp']) * household_weight
df_urban['poor_hat_685'] = (df_urban['income_pc_hat'] <= df_urban['lp_685usd_ppp']) * household_weight
df_urban['poor_hat_365'] = (df_urban['income_pc_hat'] <= df_urban['lp_365usd_ppp']) * household_weight
df_urban['poor_hat_215'] = (df_urban['income_pc_hat'] <= df_urban['lp_215usd_ppp']) * household_weight
income_series = (df_urban.groupby(['year','lima_metropolitana'])
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
income_series['date'] = pd.to_datetime(income_series[['year']].assign(MONTH=1,DAY=1))

# Split between urbano and rural:
income_series_lima = income_series.query('lima_metropolitana==1')

# Plotting:
plt.clf()
plt.figure(figsize=(10, 10))
# Plotting the means with standard deviation
# poor_685
plt.errorbar(income_series_lima['date'], income_series_lima['poor_685'], yerr=income_series_lima['std_685_mean'], 
             label='LP 685', color=settings.color1, fmt='-')
plt.errorbar(income_series_lima['date'], income_series_lima['poor_hat_685'], yerr=income_series_lima['std_685_mean'], 
             label='LP 685 Predict', color=settings.color1, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_lima['date'], income_series_lima['poor_365'], yerr=income_series_lima['std_365_mean'], 
             label='LP 365', color=settings.color3, fmt='-')
plt.errorbar(income_series_lima['date'], income_series_lima['poor_hat_365'], yerr=income_series_lima['std_365_mean'], 
             label='LP 365 Predict', color=settings.color3, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.errorbar(income_series_lima['date'], income_series_lima['poor_215'], yerr=income_series_lima['std_215_mean'], 
             label='LP 215', color=settings.color5, fmt='-')
plt.errorbar(income_series_lima['date'], income_series_lima['poor_hat_215'], yerr=income_series_lima['std_215_mean'], 
             label='LP 215 Predict', color=settings.color5, fmt='-.', linestyle='-.')  # Adjust linestyle if needed
plt.xlabel('Date')
plt.ylabel('Poverty Rate')
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig8c_poverty_rate_time_series_lima.pdf', bbox_inches='tight')

print('Figure 8c saved')

#%% Figure 9 (fig9_gini_time_series): 
# Gini Coefficient
#------------------------------------------

df['n_people'] = df['mieperho'] * df['pondera_i']
household_weight = df['n_people']/df.groupby('year')['n_people'].transform('sum')

def gini_coefficient(income_data):
    sorted_income = np.sort(income_data)
    n = len(income_data)
    cumulative_income = np.cumsum(sorted_income)
    gini = (n + 1 - 2 * np.sum(cumulative_income) / cumulative_income[-1]) / n
    return gini

income_series = df.groupby('year')['income_pc'].apply(gini_coefficient).reset_index().rename(columns={'income_pc':'gini'})

income_series['gini_hat'] = list(df.groupby('year')['income_pc_hat'].apply(gini_coefficient))

# Convert 'year' and 'month' to a datetime
income_series['date'] = pd.to_datetime(income_series[['year']].assign(MONTH=1,DAY=1))

# Plotting:
plt.clf()
# Plotting the means with standard deviation
plt.figure(figsize=(10, 10))
plt.plot(income_series['date'], income_series['gini'], label='True GINI', color=settings.color2)
plt.plot(income_series['date'], income_series['gini_hat'], label='Predicted GINI', color=settings.color4, linestyle='--')
plt.xlabel('Date')
plt.ylabel('Income')
plt.ylim(0.2, .8)
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig9_gini_time_series.pdf', bbox_inches='tight')

print('Figure 9 saved')


print('End of code: 04_generate_prediction_report.py')





