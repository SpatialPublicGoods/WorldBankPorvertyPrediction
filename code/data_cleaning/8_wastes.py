##############################################
#
# WORKING WASTES DATA
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
freq = 'y'
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
file    = os.path.join(o1_path, 'residuos/A. Generación Anual de residuos domiciliario_Distrital_2014_2021.csv')
data_df = pd.read_csv(file, encoding='iso-8859-1', on_bad_lines='skip', sep=";")

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
    'ubigeo'       : 'ubigeo',
    'pob urbana'   : 'urban_pop',
    'residuos dom' : 'wastes',
    'periodo'      : 'year'
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
    'urban_pop' : 'sum',
    'wastes'    : 'sum' 
}).reset_index()

#--------------
# Sorting data
#--------------
df = df.sort_values(by=agg_list)

#--------------
# Creating variables
#--------------
df['wastes'] = df['wastes'].str.replace(',', '.').astype(float)
df['wastes_m'] = df['wastes']/df['urban_pop']
df['wastes_m']  = df['wastes_m'].round(4)

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
export_file = os.path.join(d1_path, 'wastes.csv')
final_df.to_csv(export_file, index=False, encoding='utf-8')

# %%