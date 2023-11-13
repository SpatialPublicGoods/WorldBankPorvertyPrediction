##############################################
#
# WORKING TRANSPORT INDUSTRY DATA
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
# 1. CLEANING DATA
#
# ============================

# %%

#--------------
# Opening
#--------------
file = os.path.join(o1_path, 'Transporte carga/1_Transporte_Carga_Carretero_2022.csv')
df   = pd.read_csv(file, encoding='iso-8859-1', on_bad_lines='skip')

#--------------
# Cleaning column names
#--------------

# Dictionary of matching column names & new column names
name_mapping = {
    'ubigeo'      : 'ubigeo',
    'fab anio'    : 'fab_date',
    'servicio'    : 'service',
    'fecha corte' : 'date',
    'id'          : 'id'
}

# Map function to rename old matching names with new column names
def map_column_name(name):
    best_match, score = process.extractOne(name.lower(), name_mapping.keys())
    if score >= 80:
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
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

df['day']   = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year']  = df['date'].dt.year

df.loc[(df['month'] <= 3)                     , 'quarter'] = 1
df.loc[(df['month'] >= 4) & (df['month'] <= 6), 'quarter'] = 2
df.loc[(df['month'] >= 7) & (df['month'] <= 9), 'quarter'] = 3
df.loc[(df['month'] >= 10)                    , 'quarter'] = 4
df['quarter'] = df['quarter'].astype(int)

### Vehicles fabrication years
df['fab_years'] = df['year'] - df['fab_date']

df.loc[(df['fab_years'] <= 5) , 'fab_5y'] = 1
df.loc[(df['fab_years'] >  5) , 'fab_5y'] = 0
df.loc[(df['fab_years'] <= 10) , 'fab_10y'] = 1
df.loc[(df['fab_years'] >  10) , 'fab_10y'] = 0
df.loc[(df['fab_years'] <= 20) , 'fab_20y'] = 1
df.loc[(df['fab_years'] >  20) , 'fab_20y'] = 0
df.loc[(df['fab_years'] <= 30) , 'fab_30y'] = 1
df.loc[(df['fab_years'] >  30) , 'fab_30y'] = 0

### Service type
def is_public_service(service):
    min_score = 80
    if fuzz.partial_ratio("publico", str(service).lower()) >= min_score:
        return 1
    else:
        return 0

df['pub_serv'] = df['service'].apply(is_public_service)

#--------------
# Aggregating by ubigeo & date
#--------------

# Map for aggregation list
agg_map = {
    'd': ['ubigeo', 'year', 'quarter', 'month', 'day'],
    'm': ['ubigeo', 'year', 'quarter', 'month'],
    'q': ['ubigeo', 'year', 'quarter'],
    'y': ['ubigeo', 'year']
}

# Defining aggregation list
agg_list = agg_map.get(freq, [])

# Aggregating by aggregation list
df = df.groupby(agg_list).agg({
    'id'       : 'count',
    'fab_5y'   : 'sum',
    'fab_10y'  : 'sum',
    'fab_20y'  : 'sum',
    'fab_30y'  : 'sum',
    'pub_serv' : 'sum',
    }).reset_index()

#--------------
# Sorting data
#--------------
df = df.sort_values(by=agg_list)

#--------------
# Renaming
#--------------
df = df.rename(columns={'id': 'vehicles_tot'})

#--------------
# Percentage variables
#--------------
df['fab_5y']   = (df['fab_5y']   / df['vehicles_tot']).round(4)
df['fab_10y']  = (df['fab_10y']  / df['vehicles_tot']).round(4)
df['fab_20y']  = (df['fab_20y']  / df['vehicles_tot']).round(4)
df['fab_30y']  = (df['fab_30y']  / df['vehicles_tot']).round(4)
df['pub_serv'] = (df['pub_serv'] / df['vehicles_tot']).round(4)

# %%





# ============================
#
# 2. EXPORTING DATA
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
export_file = os.path.join(d1_path, 'cargo_vehicles.csv')
final_df.to_csv(export_file, index=False, encoding='utf-8')

# %%