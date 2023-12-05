##############################################
#
# WORKING CLUSTERS DATA
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
o1_path = os.path.join(main_path, '1_raw/peru/data')
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
file    = os.path.join(o1_path, 'conglomerado_centroidehogaresporconglomerado_2013-2021.csv')
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
    'conglome' : 'conglome',
    'ubigeo'   : 'ubigeo',
    'mes'      : 'month',
    'ano'      : 'year'
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
# Dropping duplicates
#--------------
df = df.drop_duplicates()

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
export_file = os.path.join(d1_path, 'clusters.csv')
final_df.to_csv(export_file, index=False, encoding='utf-8')

# %%