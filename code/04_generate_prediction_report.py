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



#%% Get current working directory and parameters:

# Parameters
dataPath = 'J:/My Drive/PovertyPredictionRealTime/data'

dataPath = '/home/fcalle0/datasets/WorldBankPovertyPrediction/'

freq = 'm'

date = '2023-12-12' #datetime.today().strftime('%Y-%m-%d')

#--------------

dpml = DataPreparationForML(freq=freq, dataPath=dataPath, date=date)

# 1. Read dataset:

ml_dataset = (dpml.read_consolidated_ml_dataset()
                    .groupby(['ubigeo','conglome','vivienda','hogar_ine','year'])
                    .first()
                    .reset_index(drop=False)
                    )

# 2. Obtain filtered dataset:

ml_dataset_filtered_train = dpml.filter_ml_dataset(ml_dataset).query('year<=2018')

Y_standardized_train, X_standardized_train, scaler_X_train, scaler_Y_train = dpml.get_depvar_and_features(ml_dataset_filtered_train)

ml_dataset_filtered_validation = dpml.filter_ml_dataset(ml_dataset).query('year==2019')

Y_standardized_validation, X_standardized_validation, scaler_X_validation, scaler_Y_validation = dpml.get_depvar_and_features(ml_dataset_filtered_validation, scaler_X_train, scaler_Y_train)

# 3. Load best model:

best_model = dpml.load_ml_model(model_filename = 'best_weighted_lasso_model.joblib')

# 4. Predict income

ml_dataset_filtered_validation['log_income_pc_hat'] = best_model.predict(X_standardized_validation)
ml_dataset_filtered_validation['income_pc_hat'] = np.exp(ml_dataset_filtered_validation['log_income_pc_hat'] * scaler_Y_train.scale_[0] + scaler_Y_train.mean_[0])

#%% Figure 1: Distribution of predicted income vs true income
plt.clf()
sns.histplot(ml_dataset_filtered_validation['income_pc_hat'], color='red', kde=True, label='Predicted Income', stat='density')
sns.histplot(ml_dataset_filtered_validation['income_pc'], color='blue', kde=True, label='True Income', stat='density')
plt.xlim(0, 3000)
plt.legend()
plt.savefig('../figures/fig1_prediction_vs_true_income_distribution_lasso_training_weighted.pdf', bbox_inches='tight')
plt.show()

#%% Figure 1b: ECDF of predicted income vs true income

plt.clf()
sns.ecdfplot(ml_dataset_filtered_validation['income_pc_hat'], color='red', label='Predicted Income')
sns.ecdfplot(ml_dataset_filtered_validation['income_pc'], color='blue', label='True Income')
plt.xlim(0, 2500)
plt.legend()
plt.title('ECDF of Predicted vs True Income')
plt.xlabel('Income')
plt.ylabel('Cumulative Distribution')
plt.savefig('../figures/fig1b_prediction_vs_true_income_ecdf_lasso_training_weighted.pdf', bbox_inches='tight')
plt.show()

#%% Figure 2: Distribution of predicted income vs true income by region

ml_dataset_filtered_validation['ubigeo_region'] = ml_dataset_filtered_validation['ubigeo'].str[:4]

n_rows = 5
n_cols = 5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20), sharex=True, sharey=True)

for i, region in enumerate(ml_dataset_filtered_validation['ubigeo_region'].unique()):
    ax = axes[i // n_cols, i % n_cols]
    region_data = ml_dataset_filtered_validation[ml_dataset_filtered_validation['ubigeo_region'] == region]
    sns.histplot(region_data['income_pc_hat'], color='red', kde=True, label='Predicted Income', stat='density', ax=ax)
    sns.histplot(region_data['income_pc'], color='blue', kde=True, label='True Income', stat='density', ax=ax)
    ax.set_xlim(0, 3000)
    ax.set_title(region)
    ax.legend()
    plt.savefig('../figures/fig2_prediction_vs_true_income_by_region_lasso_training_weighted.pdf', bbox_inches='tight')

#%% Figure 3: Replicate poverty rate:

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

# porverty_comparison.plot(kind='bar', figsize=(10, 6), color=['#1f77b4', '#ff7f0e'])  # Replace with your preferred colors
porverty_comparison.plot(kind='bar', figsize=(10, 6))  # Replace with your preferred colors
plt.title('Comparison of Actual vs Predicted Poverty by Line')
plt.ylabel('Sum')
plt.xlabel('Poverty Line')
plt.xticks(rotation=45)
plt.ylim(0, .5)
plt.savefig('../figures/fig3_prediction_vs_true_poverty_rate_national.pdf', bbox_inches='tight')

#%% Figure 4: Replicate poverty rate (by region)

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

plt.tight_layout()

#%% Figure 5: 

month_to_quarter = {1:1, 2:1, 3:1, 
                    4:4, 5:4, 6:4, 
                    7:7, 8:7, 9:7, 
                    10:10, 11:10, 12:10}

ml_dataset_filtered_train['quarter'] = ml_dataset_filtered_train['month'].map(month_to_quarter)
ml_dataset_filtered_validation['quarter'] = ml_dataset_filtered_validation['month'].map(month_to_quarter)

income_series_train = (ml_dataset_filtered_train.groupby(['year','quarter'])
                                                .agg({'income_pc': 'mean'})
                                                .reset_index()
                                                )

income_series_train['income_pc_hat'] = income_series_train['income_pc']

income_series_validation = (ml_dataset_filtered_validation.groupby(['year','quarter'])
                                                        .agg({'income_pc': 'mean', 'income_pc_hat': 'mean'})
                                                        .reset_index()
                                                        )

income_series = pd.concat([income_series_train, income_series_validation], axis=0).rename(columns={'quarter':'month'})

# Convert 'year' and 'month' to a datetime
income_series['date'] = pd.to_datetime(income_series[['year', 'month']].assign(DAY=1))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(income_series['date'], income_series['income_pc'], label='True Income', color='blue')
plt.plot(income_series['date'], income_series['income_pc_hat'], label='Predicted Income', color='red')
plt.xlabel('Date')
plt.ylabel('Income')
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig5_median_income_time_series.pdf', bbox_inches='tight')



#%% Figure 5: Time series Plot (Yearly)

ml_dataset_filtered_train['income_pc_hat'] = ml_dataset_filtered_train['income_pc']

df = pd.concat([ml_dataset_filtered_train, ml_dataset_filtered_validation], axis=0)

df['n_people'] = df['mieperho'] * df['pondera_i']

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
plt.figure(figsize=(10, 6))

# Plotting the means with standard deviation
plt.errorbar(income_series['date'], income_series['income_pc_weighted'], yerr=income_series['std_mean'], 
             label='True Income', color='blue', fmt='-')
plt.errorbar(income_series['date'], income_series['income_pc_hat_weighted'], yerr=income_series['std_hat_mean'], 
             label='Predicted Income', color='red', fmt='-.', linestyle='-.')  # Adjust linestyle if needed

plt.xlabel('Date')
plt.ylabel('Income')
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig5_median_income_time_series.pdf', bbox_inches='tight')
plt.show()


#%% Figure 6: Time series Plot (Quarter)

df = pd.concat([ml_dataset_filtered_train, ml_dataset_filtered_validation], axis=0)

df['n_people'] = df['mieperho'] * df['pondera_i']

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
plt.figure(figsize=(10, 6))

# Plotting the means with standard deviation
plt.errorbar(income_series['date'], income_series['income_pc_weighted'], yerr=income_series['std_mean'], 
             label='True Income', color='blue', fmt='-')
plt.errorbar(income_series['date'], income_series['income_pc_hat_weighted'], yerr=income_series['std_hat_mean'], 
             label='Predicted Income', color='red', fmt='-.', linestyle='-.')  # Adjust linestyle if needed

plt.xlabel('Date')
plt.ylabel('Income')
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig6_median_income_time_series_quarterly.pdf', bbox_inches='tight')
plt.show()


#%% Figure 7: Gini Coefficient

df = pd.concat([ml_dataset_filtered_train, ml_dataset_filtered_validation], axis=0)

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
plt.figure(figsize=(10, 6))
plt.plot(income_series['date'], income_series['gini'], label='True GINI', color='blue')
plt.plot(income_series['date'], income_series['gini_hat'], label='Predicted GINI', color='red')
plt.xlabel('Date')
plt.ylabel('Income')
plt.ylim(0.2, .8)
plt.legend()
plt.grid(True)
plt.savefig('../figures/fig7_gini_time_series.pdf', bbox_inches='tight')







