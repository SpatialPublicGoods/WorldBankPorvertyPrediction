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

# 2. Obtain filtered dataset:

year_end = 2021

ml_dataset_filtered_train = dpml.filter_ml_dataset(ml_dataset).query('year<=2016')

ml_dataset_filtered_validation = (
                                    dpml.filter_ml_dataset(ml_dataset, year_end = year_end)
                                        .query('year >= 2017')
                                        .query('year <= ' + str(year_end))
                                        .query('true_year==2016') # Keep only observations that correspond to 2016 data
                                    )

ml_dataset_filtered_true = (
                                    dpml.filter_ml_dataset(ml_dataset, year_end = year_end)
                                        .query('year >= 2017')
                                        .query('year <= ' + str(year_end))
                                        .query('true_year != 2016') # Keep only observations that correspond to 2016 data
                                    )

g_2016 = 2.4
g_2017 =	0.7
g_2018 =	2.0
g_2019 =	0.4
g_2020 =	-12.2
g_2021 =	12.0

growth_scale = lambda x: 1 + x/100


growth_rate = {2017: growth_scale(g_2017), 
                2018: growth_scale(g_2017) * growth_scale(g_2018), 
                2019: growth_scale(g_2017) * growth_scale(g_2018)  * growth_scale(g_2019),
                2020: growth_scale(g_2017) * growth_scale(g_2018)  * growth_scale(g_2019) * growth_scale(g_2020),
                2021: growth_scale(g_2017) * growth_scale(g_2018)  * growth_scale(g_2019) * growth_scale(g_2020) * growth_scale(g_2021)
                }


# 5. Predict income
ml_dataset_filtered_train['income_pc_hat'] = ml_dataset_filtered_train['income_pc']

ml_dataset_filtered_validation['income_pc_hat'] = ml_dataset_filtered_validation['income_pc'] * ml_dataset_filtered_validation['year'].map(growth_rate)


# 5. Compiling both datasets and creating some variables:

df = pd.concat([ml_dataset_filtered_train, ml_dataset_filtered_validation], axis=0)

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
sns.histplot(ml_dataset_filtered_validation['income_pc_hat'], 
                color=settings.color1, 
                # kde=True, 
                label='Predicted Income', 
                stat='density', 
                fill=False, 
                element='step'
                )
sns.histplot(ml_dataset_filtered_true['income_pc'], 
                color=settings.color2, 
                # kde=True, 
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
sns.ecdfplot(ml_dataset_filtered_validation['income_pc_hat'], color=settings.color1, label='Predicted Income')
sns.ecdfplot(ml_dataset_filtered_true['income_pc'], color=settings.color2, label='True Income')
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
    plt.savefig('../figures/baseline_report/fig2_prediction_vs_true_income_by_region_lasso_training_weighted.pdf', bbox_inches='tight')

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
porverty_comparison_diff.iloc[:,:] = (np.array(porverty_comparison_test) - np.array(porverty_comparison_pred)) / np.array(porverty_comparison_test)


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
plt.savefig('../figures/baseline_report/fig3_prediction_vs_true_poverty_rate_national.pdf', bbox_inches='tight')

print('Figure 3 saved')

#%% Figure 4 (fig4_prediction_vs_true_poverty_rate_regions): 
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



for yy in range(2017, year_end+1):
  
  plt.clf()

  n_rows = 5
  n_cols = 5
  fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))

  # Loop over each ubigeo_region
  for i, (region, data) in enumerate(porverty_comparison_region.query('year == ' + str(yy)).iterrows()):
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

  plt.savefig('../figures/baseline_report/fig4_prediction_vs_true_poverty_rate_regions' + str(yy) + '.pdf', bbox_inches='tight')

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
plt.savefig('../figures/baseline_report/fig4_2_prediction_vs_true_poverty_rate_regions_p685_scatter.pdf', bbox_inches='tight')


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
plt.savefig('../figures/baseline_report/fig4_2_prediction_vs_true_poverty_rate_regions_p365_scatter.pdf', bbox_inches='tight')


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
plt.savefig('../figures/baseline_report/fig4_2_prediction_vs_true_poverty_rate_regions_p215_scatter.pdf', bbox_inches='tight')


# get mse for regional comparison:
porverty_comparison_regional_mse = porverty_comparison_region.reset_index().copy()
porverty_comparison_regional_mse['sq_error_685'] = (porverty_comparison_regional_mse['poor_685']-porverty_comparison_regional_mse['poor_hat_685'])**2
porverty_comparison_regional_mse['sq_error_365'] = (porverty_comparison_regional_mse['poor_365']- porverty_comparison_regional_mse['poor_hat_365'])**2
porverty_comparison_regional_mse['sq_error_215'] = (porverty_comparison_regional_mse['poor_215']- porverty_comparison_regional_mse['poor_hat_215'])**2
pov_regional_mse = porverty_comparison_regional_mse.groupby('year')[['sq_error_685', 'sq_error_365', 'sq_error_215']].mean()
pov_regional_mse['type']='regional'



#%% Figure 4.1 (fig4_prediction_vs_true_poverty_rate_provincia): 
# Replicate poverty rate (by provincia)
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
plt.savefig('../figures/baseline_report/fig4_3_prediction_vs_true_poverty_rate_provincia_p685_scatter.pdf', bbox_inches='tight')




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
plt.savefig('../figures/baseline_report/fig4_2_prediction_vs_true_poverty_rate_provincia_p365_scatter.pdf', bbox_inches='tight')


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
plt.savefig('../figures/baseline_report/fig4_2_prediction_vs_true_poverty_rate_provincia_p215_scatter.pdf', bbox_inches='tight')


# get mse for provincial comparison:
porverty_comparison_provincial_mse = porverty_comparison_provincia.reset_index().copy()
porverty_comparison_provincial_mse['sq_error_685'] = (porverty_comparison_provincial_mse['poor_685']-porverty_comparison_provincial_mse['poor_hat_685'])**2
porverty_comparison_provincial_mse['sq_error_365'] = (porverty_comparison_provincial_mse['poor_365']- porverty_comparison_provincial_mse['poor_hat_365'])**2
porverty_comparison_provincial_mse['sq_error_215'] = (porverty_comparison_provincial_mse['poor_215']- porverty_comparison_provincial_mse['poor_hat_215'])**2
pov_provincial_mse = porverty_comparison_provincial_mse.groupby('year')[['sq_error_685', 'sq_error_365', 'sq_error_215']].mean()
pov_provincial_mse['type']= 'provincial'

pov_mse = pd.concat([pov_regional_mse, pov_provincial_mse], axis=0)

pov_mse.to_csv('../tables/table1_pov_mse_wb.csv')

#--------------------------------------------------------------
# Time Series Analysis

time_series_analysis = False

# if time_series_analysis:


#   #%% Figure 5 (fig5_prediction_vs_true_poverty_rate_urban_rural): 
#   # Replicate poverty rate urban-rural
#   #----------------------------------------------------

#   ml_dataset_filtered_test['n_people'] = ml_dataset_filtered_test['mieperho'] * ml_dataset_filtered_test['pondera_i']
#   household_weight_test = ml_dataset_filtered_test['n_people']/ml_dataset_filtered_test.groupby(['year', 'urbano'])['n_people'].transform('sum')
#   ml_dataset_filtered_test['poor_685'] = (ml_dataset_filtered_test['income_pc'] <= ml_dataset_filtered_test['lp_685usd_ppp']) * household_weight_test
#   ml_dataset_filtered_test['poor_365'] = (ml_dataset_filtered_test['income_pc'] <= ml_dataset_filtered_test['lp_365usd_ppp']) * household_weight_test
#   ml_dataset_filtered_test['poor_215'] = (ml_dataset_filtered_test['income_pc'] <= ml_dataset_filtered_test['lp_215usd_ppp']) * household_weight_test

#   ml_dataset_filtered_prediction['n_people'] = ml_dataset_filtered_prediction['mieperho'] * ml_dataset_filtered_prediction['pondera_i']
#   household_weight_prediction = ml_dataset_filtered_prediction['n_people']/ml_dataset_filtered_prediction.groupby(['year', 'urbano'])['n_people'].transform('sum')
#   ml_dataset_filtered_prediction['poor_hat_685'] = (ml_dataset_filtered_prediction['income_pc_hat'] <= ml_dataset_filtered_prediction['lp_685usd_ppp']) * household_weight_prediction
#   ml_dataset_filtered_prediction['poor_hat_365'] = (ml_dataset_filtered_prediction['income_pc_hat'] <= ml_dataset_filtered_prediction['lp_365usd_ppp']) * household_weight_prediction
#   ml_dataset_filtered_prediction['poor_hat_215'] = (ml_dataset_filtered_prediction['income_pc_hat'] <= ml_dataset_filtered_prediction['lp_215usd_ppp']) * household_weight_prediction


#   porverty_comparison_test = ml_dataset_filtered_test.loc[:,['urbano','poor_685','poor_365','poor_215']].groupby('urbano').sum()

#   porverty_comparison_pred = ml_dataset_filtered_prediction.loc[:,['urbano','poor_hat_685','poor_hat_365','poor_hat_215']].groupby('urbano').sum()

#   porverty_comparison_region = pd.concat([porverty_comparison_test, porverty_comparison_pred], axis=1)


#   plt.clf()

#   n_rows = 2
#   n_cols = 2
#   fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))

#   # Loop over each ubigeo_region
#   for i, (region, data) in enumerate(porverty_comparison_region.iterrows()):
#       #
#       data = data.reset_index()
#       data.columns = ['index', 'PovertyRate']
#       data[['Poverty Line', 'Type']] = data['index'].str.rsplit('_', n=1, expand=True)
#       data = data.pivot(index='Poverty Line', columns='Type', values='PovertyRate')
#       #
#       ax = axes[i // n_cols, i % n_cols]
#       data.plot(kind='bar', ax=ax)
#       ax.set_title(region)
#       ax.set_ylabel('Rate')
#       ax.set_xlabel('Poverty Line')
#       ax.set_ylim(0, .8)
#       plt.xticks(rotation=90)

#   # Hide unused subplots
#   for j in range(i + 1, n_rows * n_cols):
#       axes[j // n_cols, j % n_cols].axis('off')

#   plt.savefig('../figures/baseline_report/fig5_prediction_vs_true_poverty_rate_urban_rural.pdf', bbox_inches='tight')

#   print('Figure 5 saved')

#   #%% Figure 7 (fig7_diff_nacional): 
#   # Time series average income plot by area (Yearly)
#   #----------------------------------------------------

#   ml_dataset_filtered_test['n_people'] = ml_dataset_filtered_test['mieperho'] * ml_dataset_filtered_test['pondera_i']
#   household_weight_test = ml_dataset_filtered_test['n_people']/ml_dataset_filtered_test.groupby('year')['n_people'].transform('sum')
#   ml_dataset_filtered_test['poor_685'] = (ml_dataset_filtered_test['income_pc'] <= ml_dataset_filtered_test['lp_685usd_ppp']) * household_weight_test
#   ml_dataset_filtered_test['poor_365'] = (ml_dataset_filtered_test['income_pc'] <= ml_dataset_filtered_test['lp_365usd_ppp']) * household_weight_test
#   ml_dataset_filtered_test['poor_215'] = (ml_dataset_filtered_test['income_pc'] <= ml_dataset_filtered_test['lp_215usd_ppp']) * household_weight_test

#   ml_dataset_filtered_prediction['n_people'] = ml_dataset_filtered_prediction['mieperho'] * ml_dataset_filtered_prediction['pondera_i']
#   household_weight_prediction = ml_dataset_filtered_prediction['n_people']/ml_dataset_filtered_prediction.groupby('year')['n_people'].transform('sum')
#   ml_dataset_filtered_prediction['poor_hat_685'] = (ml_dataset_filtered_prediction['income_pc_hat'] <= ml_dataset_filtered_prediction['lp_685usd_ppp']) * household_weight_prediction
#   ml_dataset_filtered_prediction['poor_hat_365'] = (ml_dataset_filtered_prediction['income_pc_hat'] <= ml_dataset_filtered_prediction['lp_365usd_ppp']) * household_weight_prediction
#   ml_dataset_filtered_prediction['poor_hat_215'] = (ml_dataset_filtered_prediction['income_pc_hat'] <= ml_dataset_filtered_prediction['lp_215usd_ppp']) * household_weight_prediction

#   porverty_comparison_test = ml_dataset_filtered_test.loc[:,['poor_685','poor_365','poor_215']].sum()

#   porverty_comparison_pred = ml_dataset_filtered_prediction.loc[:,['poor_hat_685','poor_hat_365','poor_hat_215']].sum()

#   diff_national = porverty_comparison_pred.values - porverty_comparison_test.values

