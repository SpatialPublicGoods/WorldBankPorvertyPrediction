##############################################
#
# ADMINISTRATIVE DATA DESCRIPTIVE STATISTICS
#
##############################################










# ============================
#
# 0. SETTINGS
#
# ============================

#%%

#--------------
# Libraries
#--------------
import os
import pandas as pd
from unidecode import unidecode
import numpy as np

#--------------
# Paths
#--------------
main_path = 'L:/.shortcut-targets-by-id/12-fuK40uOBz3FM-OXbZtk_OSlYcGpzpa/PovertyPredictionRealTime'
o1_path = os.path.join(main_path, 'data/4_clean')
d1_path = os.path.join(main_path, 'paper/tables/descriptive_statistics')

#%%










# ============================
#
# 1. OPENING DATA
#
# ============================

# %%

#--------------
# Opening main data
#--------------
file_name = os.path.join(o1_path, 'ml_dataset_2024-04-24.csv')
df1       = pd.read_csv(file_name, encoding='iso-8859-1', on_bad_lines='skip')

# %%










# ============================
#
# 2. CREATING SUBDATASETS
#
# ============================

# %%

#--------------
# Copy
#--------------
df2 = df1.copy()

#--------------
# New dataframes
#--------------

### Lists for admin data ##############
list_police    = ['Economic_Commercial_Offenses',
                  'Family_Domestic_Issues', 'Fraud_Financial_Crimes',
                  'Information_Cyber_Crimes', 'Intellectual_Property_Cultural_Heritage',
                  'Miscellaneous_Offenses', 'Personal_Liberty_Violations',
                  'Property_Real_Estate_Crimes', 'Public_Administration_Offenses',
                  'Public_Order_Political_Crimes', 'Public_Safety_Health',
                  'Sexual_Offenses', 'Theft_Robbery_Related_Crimes', 'Violence_Homicide']
list_vehicles  = ['vehicles_tot', 'fab_5y_p', 'fab_10y_p', 'fab_20y_p',
                  'fab_30y_p', 'pub_serv_p', 'payload_m', 'dry_weight_m',
                  'gross_weight_m', 'length_m', 'width_m', 'height_m']
list_planilla  = ['companies', 'workers_tot', 'workers_sex_f_perc', 'workers_sex_m_perc', 
                  'workers_sex_si_perc', 'workers_type_exe_perc', 'workers_type_wor_perc', 
                  'workers_type_emp_perc', 'workers_type_nd_perc', 'salaries_mean']
list_pubincome = ['canon', 'foncomun', 'impuestos_municipales', 'recursos_directamente_recaudados']
list_pubexpend = ['pubexp_dev_agriculture',	
                  'pubexp_dev_commerce'	,
                  'pubexp_dev_communications',	
                  'pubexp_dev_culture_sports',	
                  'pubexp_dev_education'	,
                  'pubexp_dev_energy'	,
                  'pubexp_dev_environment',	
                  'pubexp_dev_fishing'	,
                  'pubexp_dev_health'	,
                  'pubexp_dev_industry'	,
                  'pubexp_dev_justice'	,
                  'pubexp_dev_labor'	,
                  'pubexp_dev_mining'	,
                  'pubexp_dev_na' ,
                  'pubexp_dev_planning_management' ,
                  'pubexp_dev_public_debt' ,
                  'pubexp_dev_public_order' ,
                  'pubexp_dev_sanitation' ,
                  'pubexp_dev_social_protection' ,
                  'pubexp_dev_social_security' ,
                  'pubexp_dev_tourism' ,
                  'pubexp_dev_transport',	
                  'pubexp_dev_urban_development']
list_admin_data = ['year'] + list_police + list_vehicles + list_planilla + list_pubincome + list_pubexpend

### List for ENAHO data ###############
list_depvar    = ['log_income_pc']
list_enaho     = ['log_income_pc_lagged', 'log_income_pc_lagged2', 'log_income_pc_lagged3', 'log_income_pc_lagged4']
list_indiv_var = ['nro_hijos', 'prii', 'seci', 'secc', 'supi', 'supc', 'edad', 'male', 'informal']
list_enaho_data = ['year'] + list_depvar + list_enaho + list_indiv_var

### List for weather and geo data #####
list_precipitation    = ['Std_precipitation']
list_temperature_max  = ['Std_temperature_max']
list_temperature_min  = ['Std_temperature_min']
list_nightlights      = ['min_nightlight',
                         'max_nightlight',
                         'mean_nightlight',
                         'stdDev_nightlight',
                         'median_nightlight',
                         'range_nightlight']
list_geo_data = ['year'] + list_precipitation + list_temperature_max + list_temperature_min + list_nightlights

### Dataframes ########################
df3_admin_data = df2[list_admin_data]
df3_enaho_data = df2[list_enaho_data] 
df3_geo_data   = df2[list_geo_data]

#--------------
# Defining year intervals
#--------------
intervals = {
    '2010-2016': (2010, 2016),
    '2017':      (2017, 2017),
    '2018':      (2018, 2018),
    '2019':      (2019, 2019),
    '2020':      (2020, 2020),
    '2021':      (2021, 2021)
}

#--------------
# Calculations function
#--------------
def stats_calc(df):

    ### Empty dataframe
    result_list = []
    
    ### Analysis algorithm
    for col in df.columns[df.columns != 'year']:
        stats_dict = {'variables': col}
        for period, (start, end) in intervals.items():
            subset = df[(df['year'] >= start) & (df['year'] <= end)][col]
            stats_dict[f'mean_{period}'] = round(subset.mean(), 3)
            stats_dict[f'std_{period}'] = round(subset.std(), 3)
            stats_dict[f'count_{period}'] = subset.count()
        result_list.append(stats_dict)

    # Final dataframe
    result_df = pd.DataFrame(result_list)
    return result_df

#--------------
# Applying function
#--------------
result_admin_data = stats_calc(df3_admin_data)
result_enaho_data = stats_calc(df3_enaho_data)
result_geo_data   = stats_calc(df3_geo_data)

# %%










# ============================
#
# 3. EXPORTING TABLES
#
# ============================

# %%

#--------------
# Copy
#--------------
final_admin_data = result_admin_data.copy()
final_enaho_data = result_enaho_data.copy()
final_geo_data   = result_geo_data.copy()

#--------------
# Exporting tables
#--------------
export_file1 = os.path.join(d1_path, 'descriptives_admin_data.csv')
export_file2 = os.path.join(d1_path, 'descriptives_enaho_data.csv')
export_file3 = os.path.join(d1_path, 'descriptives_weather_geo_data.csv')

final_admin_data.to_csv(export_file1, index=False, encoding='utf-8')
final_enaho_data.to_csv(export_file2, index=False, encoding='utf-8')
final_geo_data.to_csv(export_file3, index=False, encoding='utf-8')

# %%