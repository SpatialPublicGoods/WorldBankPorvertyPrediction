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

date = '2024-02-03' #datetime.today().strftime('%Y-%m-%d')

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

ml_dataset_filtered_prediction = dpml.filter_ml_dataset(ml_dataset).query('year==2018')

ml_dataset_filtered_test = dpml.filter_ml_dataset(ml_dataset).query('year==2019')


# 5. Predict income
ml_dataset_filtered_train['income_pc_hat'] = ml_dataset_filtered_train['income_pc']

ml_dataset_filtered_prediction['income_pc_hat'] = ml_dataset_filtered_prediction['income_pc'] * (1.04) 



# 5. Compiling both datasets and creating some variables:

df = pd.concat([ml_dataset_filtered_train, ml_dataset_filtered_prediction], axis=0)

# df['income_pc_hat'] = df['income_pc_hat'] * 1.2

month_to_quarter = {1:1, 2:1, 3:1, 
                    4:4, 5:4, 6:4, 
                    7:7, 8:7, 9:7, 
                    10:10, 11:10, 12:10}

df['quarter'] = df['month'].map(month_to_quarter)

df['n_people'] = 1
df['n_people'] = df['mieperho'] * df['pondera_i']



#%% Figure 1 (fig1_prediction_vs_true_income_distribution_lasso_training_weighted): 
# Distribution of predicted income vs true income
#--------------------------------------------------------------

plt.clf()
plt.figure(figsize=(10, 10))
sns.histplot(ml_dataset_filtered_prediction['income_pc_hat'], 
                color=settings.color1, kde=True, 
                label='Predicted Income', 
                stat='density', 
                fill=False, 
                element='step'
                )
sns.histplot(ml_dataset_filtered_test['income_pc'], 
                color=settings.color2, 
                kde=True, 
                label='True Income', 
                stat='density', 
                fill=False, 
                element='step'
                )
plt.xlim(0, 3000)
plt.legend()
plt.savefig('../figures/baseline_report/fig1_prediction_vs_true_income_distribution.pdf', bbox_inches='tight')
print('Figure 1 saved')

#%% Figure 1b (fig1b_prediction_vs_true_income_ecdf_lasso_training_weighted): 
# ECDF of predicted income vs true income
#-------------------------------------------------------

plt.clf()
plt.figure(figsize=(10, 10))
sns.ecdfplot(ml_dataset_filtered_prediction['income_pc_hat'], color=settings.color1, label='Predicted Income')
sns.ecdfplot(ml_dataset_filtered_test['income_pc'], color=settings.color2, label='True Income')
plt.xlim(0, 2500)
plt.legend()
plt.xlabel('Income')
plt.ylabel('Cumulative Distribution')
plt.savefig('../figures/baseline_report/fig1b_prediction_vs_true_income_ecdf.pdf', bbox_inches='tight')
print('Figure 1b saved')

#%% Figure 2 (fig2_prediction_vs_true_income_by_region_lasso_training_weighted): 
# Distribution of predicted income vs true income by region
#---------------------------------------------------------------------------------

plt.clf()

n_rows = 5
n_cols = 5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20), sharex=True, sharey=True)

for i, region in enumerate(ml_dataset_filtered_prediction['ubigeo_region'].unique()):
    ax = axes[i // n_cols, i % n_cols]
    region_data_predicted = ml_dataset_filtered_prediction[ml_dataset_filtered_prediction['ubigeo_region'] == region]
    region_data_true = ml_dataset_filtered_test[ml_dataset_filtered_test['ubigeo_region'] == region]
    sns.histplot(region_data_predicted['income_pc_hat'], 
                    color=settings.color1, 
                    kde=True, 
                    label='Predicted Income', 
                    stat='density', 
                    fill=False, 
                    element='step',
                    ax=ax)
    sns.histplot(region_data_true['income_pc'], 
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
    plt.savefig('../figures/baseline_report/fig2_prediction_vs_true_income_by_region.pdf', bbox_inches='tight')

print('Figure 2 saved')

#%% Figure 3 (fig3_prediction_vs_true_poverty_rate_national): 
# Poverty Rate National 
#-----------------------------------------------------------------------------------

ml_dataset_filtered_test['n_people'] = ml_dataset_filtered_test['mieperho'] * ml_dataset_filtered_test['pondera_i']
household_weight_test = ml_dataset_filtered_test['n_people']/ml_dataset_filtered_test.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_test['poor_685'] = (ml_dataset_filtered_test['income_pc'] <= ml_dataset_filtered_test['lp_685usd_ppp']) * household_weight_test
ml_dataset_filtered_test['poor_365'] = (ml_dataset_filtered_test['income_pc'] <= ml_dataset_filtered_test['lp_365usd_ppp']) * household_weight_test
ml_dataset_filtered_test['poor_215'] = (ml_dataset_filtered_test['income_pc'] <= ml_dataset_filtered_test['lp_215usd_ppp']) * household_weight_test

ml_dataset_filtered_prediction['n_people'] = ml_dataset_filtered_prediction['mieperho'] * ml_dataset_filtered_prediction['pondera_i']
household_weight_prediction = ml_dataset_filtered_prediction['n_people']/ml_dataset_filtered_prediction.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_prediction['poor_hat_685'] = (ml_dataset_filtered_prediction['income_pc_hat'] <= ml_dataset_filtered_prediction['lp_685usd_ppp']) * household_weight_prediction
ml_dataset_filtered_prediction['poor_hat_365'] = (ml_dataset_filtered_prediction['income_pc_hat'] <= ml_dataset_filtered_prediction['lp_365usd_ppp']) * household_weight_prediction
ml_dataset_filtered_prediction['poor_hat_215'] = (ml_dataset_filtered_prediction['income_pc_hat'] <= ml_dataset_filtered_prediction['lp_215usd_ppp']) * household_weight_prediction

porverty_comparison_test = ml_dataset_filtered_test.loc[:,['poor_685','poor_365','poor_215']].sum()

porverty_comparison_pred = ml_dataset_filtered_prediction.loc[:,['poor_hat_685','poor_hat_365','poor_hat_215']].sum()

porverty_comparison = pd.concat([porverty_comparison_test, porverty_comparison_pred], axis=0)


porverty_comparison = pd.DataFrame(porverty_comparison).rename(columns={0:'PovertyRate'}).reset_index()

porverty_comparison[['Poverty Line', 'Type']] = porverty_comparison['index'].str.rsplit('_', n=1, expand=True)

porverty_comparison = porverty_comparison.pivot(index='Poverty Line', columns='Type', values='PovertyRate')


plt.clf()
porverty_comparison.plot(kind='bar', figsize=(10, 10), color=[settings.color1, settings.color2]) 
plt.ylabel('Sum')
plt.xlabel('Poverty Line')
plt.xticks(rotation=45)
plt.ylim(0, .5)
plt.savefig('../figures/baseline_report/fig3_prediction_vs_true_poverty_rate_national.pdf', bbox_inches='tight')

print('Figure 3 saved')

#%% Figure 4 (fig4_prediction_vs_true_poverty_rate_regions): 
# Replicate poverty rate (by region)
#----------------------------------------------------

ml_dataset_filtered_test['n_people'] = ml_dataset_filtered_test['mieperho'] * ml_dataset_filtered_test['pondera_i']
household_weight_test = ml_dataset_filtered_test['n_people']/ml_dataset_filtered_test.groupby(['year', 'ubigeo_region'])['n_people'].transform('sum')
ml_dataset_filtered_test['poor_685'] = (ml_dataset_filtered_test['income_pc'] <= ml_dataset_filtered_test['lp_685usd_ppp']) * household_weight_test
ml_dataset_filtered_test['poor_365'] = (ml_dataset_filtered_test['income_pc'] <= ml_dataset_filtered_test['lp_365usd_ppp']) * household_weight_test
ml_dataset_filtered_test['poor_215'] = (ml_dataset_filtered_test['income_pc'] <= ml_dataset_filtered_test['lp_215usd_ppp']) * household_weight_test

ml_dataset_filtered_prediction['n_people'] = ml_dataset_filtered_prediction['mieperho'] * ml_dataset_filtered_prediction['pondera_i']
household_weight_prediction = ml_dataset_filtered_prediction['n_people']/ml_dataset_filtered_prediction.groupby(['year', 'ubigeo_region'])['n_people'].transform('sum')
ml_dataset_filtered_prediction['poor_hat_685'] = (ml_dataset_filtered_prediction['income_pc_hat'] <= ml_dataset_filtered_prediction['lp_685usd_ppp']) * household_weight_prediction
ml_dataset_filtered_prediction['poor_hat_365'] = (ml_dataset_filtered_prediction['income_pc_hat'] <= ml_dataset_filtered_prediction['lp_365usd_ppp']) * household_weight_prediction
ml_dataset_filtered_prediction['poor_hat_215'] = (ml_dataset_filtered_prediction['income_pc_hat'] <= ml_dataset_filtered_prediction['lp_215usd_ppp']) * household_weight_prediction


porverty_comparison_test = ml_dataset_filtered_test.loc[:,['ubigeo_region','poor_685','poor_365','poor_215']].groupby('ubigeo_region').sum()

porverty_comparison_pred = ml_dataset_filtered_prediction.loc[:,['ubigeo_region','poor_hat_685','poor_hat_365','poor_hat_215']].groupby('ubigeo_region').sum()

porverty_comparison_region = pd.concat([porverty_comparison_test, porverty_comparison_pred], axis=1)


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

plt.savefig('../figures/baseline_report/fig4_prediction_vs_true_poverty_rate_regions.pdf', bbox_inches='tight')

print('Figure 4 saved')


#%% Figure 5 (fig5_prediction_vs_true_poverty_rate_urban_rural): 
# Replicate poverty rate urban-rural
#----------------------------------------------------

ml_dataset_filtered_test['n_people'] = ml_dataset_filtered_test['mieperho'] * ml_dataset_filtered_test['pondera_i']
household_weight_test = ml_dataset_filtered_test['n_people']/ml_dataset_filtered_test.groupby(['year', 'urbano'])['n_people'].transform('sum')
ml_dataset_filtered_test['poor_685'] = (ml_dataset_filtered_test['income_pc'] <= ml_dataset_filtered_test['lp_685usd_ppp']) * household_weight_test
ml_dataset_filtered_test['poor_365'] = (ml_dataset_filtered_test['income_pc'] <= ml_dataset_filtered_test['lp_365usd_ppp']) * household_weight_test
ml_dataset_filtered_test['poor_215'] = (ml_dataset_filtered_test['income_pc'] <= ml_dataset_filtered_test['lp_215usd_ppp']) * household_weight_test

ml_dataset_filtered_prediction['n_people'] = ml_dataset_filtered_prediction['mieperho'] * ml_dataset_filtered_prediction['pondera_i']
household_weight_prediction = ml_dataset_filtered_prediction['n_people']/ml_dataset_filtered_prediction.groupby(['year', 'urbano'])['n_people'].transform('sum')
ml_dataset_filtered_prediction['poor_hat_685'] = (ml_dataset_filtered_prediction['income_pc_hat'] <= ml_dataset_filtered_prediction['lp_685usd_ppp']) * household_weight_prediction
ml_dataset_filtered_prediction['poor_hat_365'] = (ml_dataset_filtered_prediction['income_pc_hat'] <= ml_dataset_filtered_prediction['lp_365usd_ppp']) * household_weight_prediction
ml_dataset_filtered_prediction['poor_hat_215'] = (ml_dataset_filtered_prediction['income_pc_hat'] <= ml_dataset_filtered_prediction['lp_215usd_ppp']) * household_weight_prediction


porverty_comparison_test = ml_dataset_filtered_test.loc[:,['urbano','poor_685','poor_365','poor_215']].groupby('urbano').sum()

porverty_comparison_pred = ml_dataset_filtered_prediction.loc[:,['urbano','poor_hat_685','poor_hat_365','poor_hat_215']].groupby('urbano').sum()

porverty_comparison_region = pd.concat([porverty_comparison_test, porverty_comparison_pred], axis=1)


plt.clf()

n_rows = 2
n_cols = 2
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

plt.savefig('../figures/baseline_report/fig5_prediction_vs_true_poverty_rate_urban_rural.pdf', bbox_inches='tight')

print('Figure 5 saved')

#%% Figure 7 (fig7_diff_nacional): 
# Time series average income plot by area (Yearly)
#----------------------------------------------------

ml_dataset_filtered_test['n_people'] = ml_dataset_filtered_test['mieperho'] * ml_dataset_filtered_test['pondera_i']
household_weight_test = ml_dataset_filtered_test['n_people']/ml_dataset_filtered_test.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_test['poor_685'] = (ml_dataset_filtered_test['income_pc'] <= ml_dataset_filtered_test['lp_685usd_ppp']) * household_weight_test
ml_dataset_filtered_test['poor_365'] = (ml_dataset_filtered_test['income_pc'] <= ml_dataset_filtered_test['lp_365usd_ppp']) * household_weight_test
ml_dataset_filtered_test['poor_215'] = (ml_dataset_filtered_test['income_pc'] <= ml_dataset_filtered_test['lp_215usd_ppp']) * household_weight_test

ml_dataset_filtered_prediction['n_people'] = ml_dataset_filtered_prediction['mieperho'] * ml_dataset_filtered_prediction['pondera_i']
household_weight_prediction = ml_dataset_filtered_prediction['n_people']/ml_dataset_filtered_prediction.groupby('year')['n_people'].transform('sum')
ml_dataset_filtered_prediction['poor_hat_685'] = (ml_dataset_filtered_prediction['income_pc_hat'] <= ml_dataset_filtered_prediction['lp_685usd_ppp']) * household_weight_prediction
ml_dataset_filtered_prediction['poor_hat_365'] = (ml_dataset_filtered_prediction['income_pc_hat'] <= ml_dataset_filtered_prediction['lp_365usd_ppp']) * household_weight_prediction
ml_dataset_filtered_prediction['poor_hat_215'] = (ml_dataset_filtered_prediction['income_pc_hat'] <= ml_dataset_filtered_prediction['lp_215usd_ppp']) * household_weight_prediction

porverty_comparison_test = ml_dataset_filtered_test.loc[:,['poor_685','poor_365','poor_215']].sum()

porverty_comparison_pred = ml_dataset_filtered_prediction.loc[:,['poor_hat_685','poor_hat_365','poor_hat_215']].sum()

diff_national = porverty_comparison_pred.values - porverty_comparison_test.values

