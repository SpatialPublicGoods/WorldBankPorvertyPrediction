##############################################
#
# WORKING HOSPITALS DATA
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
file    = os.path.join(o1_path, 'hospitales/IPRESS.csv')
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
    'institucion'        : 'type',
    'ubigeo'             : 'ubigeo',
    'categoria'          : 'category',
    'incio de actividad' : 'date'
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

# Private establishment
cad = "privado"
var = 'type'
new = 'private'
df[new] = df[var].apply(lambda x: is_private(x, cad))

# Category I
cad = "i-"
var = 'category'
new = 'type_i'
df[new] = df[var].apply(lambda x: is_private(x, cad))

# Category II
cad = "ii-"
var = 'category'
new = 'type_ii'
df[new] = df[var].apply(lambda x: is_private(x, cad))

# Category III
cad = "iii-"
var = 'category'
new = 'type_iii'
df[new] = df[var].apply(lambda x: is_private(x, cad))

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
    'type'     : 'count',
    'private'  : 'mean',
    'type_i'   : 'mean',
    'type_ii'  : 'mean',
    'type_iii' : 'mean' 
}).reset_index()

#--------------
# Sorting data
#--------------
df = df.sort_values(by=agg_list)

#--------------
# Sorting data
#--------------
df = df.sort_values(by=agg_list)

#--------------
# Rounding variables
#--------------
df['private']  = df['private'].round(4)
df['type_i']   = df['type_i'].round(4)
df['type_ii']  = df['type_ii'].round(4)
df['type_iii'] = df['type_iii'].round(4)

#--------------
# Renaming
#--------------
df = df.rename(columns={
    'type'     : 'hospital_tot',
    'private'  : 'private_p',
    'type_i'   : 'type_i_p',
    'type_ii'  : 'type_ii_p',
    'type_iii' : 'type_iii_p',
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
export_file = os.path.join(d1_path, 'hospitals.csv')
final_df.to_csv(export_file, index=False, encoding='utf-8')

# %%