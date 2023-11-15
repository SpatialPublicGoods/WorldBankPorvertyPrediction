##############################################
#
# WORKING CARGO VEHICLES DATA
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
# 1. OPENING DATA
#
# ============================

# %%

#--------------
# Opening main data
#--------------
file    = os.path.join(o1_path, 'Transporte carga/1_Transporte_Carga_Carretero_2022.csv')
data_df = pd.read_csv(file, encoding='iso-8859-1', on_bad_lines='skip')

#--------------
# Opening SUNAT data
#--------------
file     = os.path.join(main_path, 'PadronRUC.csv')
sunat_df = pd.read_csv(file, encoding='iso-8859-1', on_bad_lines='skip')

#--------------
# Merging data
#--------------
merged_df = data_df.merge(sunat_df, on='RUC', how='left', suffixes=('_main', '_sunat'))

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
df = merged_df.copy()

#--------------
# Cleaning column names
#--------------

# Dictionary of matching column names & new column names
name_mapping = {
    'ubigeo sunat'     : 'ubigeo',
    'fab anio'         : 'fab_date',
    'servicio'         : 'service',
    'id'               : 'id',
    'fecha resolucion' : 'date',
    'carga util'       : 'payload',
    'p seco'           : 'dry_weight',
    'p bruto'          : 'gross_weight',
    'largo'            : 'length',
    'ancho'            : 'width',
    'alto'             : 'height'
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
    'id'           : 'count',
    'fab_5y'       : 'mean',
    'fab_10y'      : 'mean',
    'fab_20y'      : 'mean',
    'fab_30y'      : 'mean',
    'pub_serv'     : 'mean',
    'payload'      : 'mean',
    'dry_weight'   : 'mean',
    'gross_weight' : 'mean',
    'length'       : 'mean',
    'width'        : 'mean',
    'height'       : 'mean'
}).reset_index()

#--------------
# Sorting data
#--------------
df = df.sort_values(by=agg_list)

#--------------
# Rounding variables
#--------------
df['fab_5y']       = df['fab_5y'].round(4)
df['fab_10y']      = df['fab_10y'].round(4)
df['fab_20y']      = df['fab_20y'].round(4)
df['fab_30y']      = df['fab_30y'].round(4)
df['pub_serv']     = df['pub_serv'].round(4)

df['payload']      = df['payload'].round(2)
df['dry_weight']   = df['dry_weight'].round(2)
df['gross_weight'] = df['length'].round(2)
df['length']       = df['length'].round(2)
df['width']        = df['width'].round(2)
df['height']       = df['height'].round(2)

#--------------
# Renaming
#--------------
df = df.rename(columns={
    'id'           : 'vehicles_tot',
    'fab_5y'       : 'fab_5y_p',
    'fab_10y'      : 'fab_10y_p',
    'fab_20y'      : 'fab_20y_p',
    'fab_30y'      : 'fab_30y_p',
    'pub_serv'     : 'pub_serv_p',
    'payload'      : 'payload_m',
    'dry_weight'   : 'dry_weight_m',
    'gross_weight' : 'gross_weight_m',
    'length'       : 'length_m',
    'width'        : 'width_m',
    'height'       : 'height_m'
    })

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
export_file = os.path.join(d1_path, 'cargo_vehicles.csv')
final_df.to_csv(export_file, index=False, encoding='utf-8')

# %%