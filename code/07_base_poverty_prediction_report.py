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

date = '2024-03-28' #datetime.today().strftime('%Y-%m-%d')

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

def compute_predicted_income_world_bank(data):

    data['income_pc_hat'] = data['income_pc'] * data['year'].map(growth_rate)

    return data

ml_dataset_filtered_validation = compute_predicted_income_world_bank(ml_dataset_filtered_validation)


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
porverty_comparison_diff.iloc[:,:] = (np.array(porverty_comparison_test) - np.array(porverty_comparison_pred))


# Plotting
plt.clf()
ax = porverty_comparison_diff.plot.bar(figsize=(10, 6), width=0.8)
ax.set_xlabel('Poverty Threshold')
ax.set_ylabel('Difference: True - Predicted')
ax.set_title('Poverty Comparison by Year')
ax.set_xticklabels(porverty_comparison_diff.index, rotation=45)
plt.legend(title='Year',  loc='upper right')
plt.ylim([-.2, .2])
plt.tight_layout()
plt.grid(True)
# plt.show()
plt.savefig('../figures/baseline_report/fig3_prediction_vs_true_poverty_rate_national.pdf', bbox_inches='tight')

print('Figure 3 saved')
