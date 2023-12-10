##############################################
#
# WORKING MORTALITY DATA
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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from fuzzywuzzy import fuzz, process
from modules.utils_general import utils_general

#--------------
# Paths
#--------------
main_path = 'L:/.shortcut-targets-by-id/12-fuK40uOBz3FM-OXbZtk_OSlYcGpzpa/PovertyPredictionRealTime/data'
o1_path = os.path.join(main_path, '1_raw/peru/big_data/admin')
d1_path = os.path.join(main_path, '2_intermediate')

#--------------
# Parameters
#--------------
freq = 'm'
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
file    = os.path.join(o1_path, 'mortalidad/TB_SINADEF.csv')
data_df = pd.read_csv(file, encoding='iso-8859-1', on_bad_lines='skip')

# %%





# ============================
#
# 2. CLEANING DATA
#
# ============================

# %%

#--------------
# Copy
#--------------
df = data_df.copy()

#--------------
# Cleaning column names
#--------------

# Dictionary of matching column names & new column names
name_mapping = {
    'fecha'                  : 'date',
    'departamento domicilio' : 'department',
    'provincia domicilio'    : 'province',
    'distrito domicilio'     : 'district',
    'sexo'                   : 'sex',
    'muerte violenta'        : 'violent_death'
}

# Map function to rename old matching names with new column names
def map_column_name(name):
    best_match, score = process.extractOne(name.lower(), name_mapping.keys())
    if score >= 90:
        return name_mapping[best_match]
    return name

# Applying the map function to rename variables based on the dictionary
df = df.rename(columns=map_column_name)

#--------------
# Filtering columns
#--------------
df = df[list(name_mapping.values())]

#--------------
# Creating variables
#--------------

### Dates
df = df[df['date'].astype(str).str.len().isin([9, 10])]  # filtering strange dates
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
df.dropna(subset=['date'], inplace=True)

df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

df.loc[(df['month'] <= 3)                     , 'quarter'] = 1
df.loc[(df['month'] >= 4) & (df['month'] <= 6), 'quarter'] = 2
df.loc[(df['month'] >= 7) & (df['month'] <= 9), 'quarter'] = 3
df.loc[(df['month'] >= 10)                    , 'quarter'] = 4
df['quarter'] = df['quarter'].astype(int)

### Dummy variables

# Defining dummy function
def is_private(variable, cadena):
    min_score = 80
    if fuzz.partial_ratio(cadena, str(variable).lower()) >= min_score:
        return 1
    else:
        return 0
    
# female
cad = "f"
var = 'sex'
new = 'female'
df[new] = df[var].apply(lambda x: is_private(x, cad))

# male
cad = "m"
var = 'sex'
new = 'male'
df[new] = df[var].apply(lambda x: is_private(x, cad))

# suicide
cad = "suicidio"
var = 'violent_death'
new = 'suicide'
df[new] = df[var].apply(lambda x: is_private(x, cad))

# traffic accident
cad = "accidente de transito"
var = 'violent_death'
new = 'traffic_accident'
df[new] = df[var].apply(lambda x: is_private(x, cad))

# murder
cad = "homicidio"
var = 'violent_death'
new = 'murder'
df[new] = df[var].apply(lambda x: is_private(x, cad))

# feminicide
df.loc[  ((df['female']==1)&(df['murder']==1)), 'feminicide'] = 1
df.loc[ ~((df['female']==1)&(df['murder']==1)), 'feminicide'] = 0
df['feminicide'] = df['feminicide'].astype(int)

#--------------
# Aggregating by ubigeo & date
#--------------

# Map for aggregation list
agg_map = {
    'd': ['department', 'province', 'district', 'year', 'quarter', 'month', 'day'],
    'm': ['department', 'province', 'district', 'year', 'quarter', 'month'],
    'q': ['department', 'province', 'district', 'year', 'quarter'],
    'y': ['department', 'province', 'district', 'year']
}

# Defining aggregation list
agg_list = agg_map.get(freq, [])

# Aggregating by aggregation list
df = df.groupby(agg_list).agg({
    'date'       : 'count',
    'female'     : 'mean',
    'male'       : 'mean',
    'suicide'    : 'mean',
    'traffic_accident'   : 'mean',
    'murder'     : 'mean',
    'feminicide' : 'mean',
}).reset_index()

#--------------
# Sorting data
#--------------
df = df.sort_values(by=agg_list)

#--------------
# Renaming
#--------------
df = df.rename(columns={
    'department' : 'region',
    'province'   : 'privincia',
    'district'   : 'distrito',
    'date'       : 'deaths_tot',
    'female'     : 'female_p',
    'male'       : 'male_p',
    'suicide'    : 'suicide_p',
    'traffic_accident'  : 'traffic_accident_p',
    'murder'     : 'murder_p',
    'feminicide' : 'feminicide_p'
    })

#--------------
# Rounding variables
#--------------
df['female_p']  = df['female_p'].round(4)
df['male_p']    = df['male_p'].round(4)

# %%








# %%
df2 = df.copy()

#--------------
# Recovering ubigeo
#--------------

utils = utils_general()

df2 = utils.input_ubigeo_to_dataframe(df2, ['region', 'provincia', 'distrito'])










# %%





# ============================
#
# 3. EXPORTING DATA
#
# ============================

# %%

#--------------
# Copy
#--------------
final_df = df.copy()

#--------------
# Exporting final dataframe
#--------------
export_file = os.path.join(d1_path, 'hospitals.csv')
final_df.to_csv(export_file, index=False, encoding='utf-8')

# %%