##############################################
#
# WORKING SENIOR NUTRITION DATA
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
import numpy as np
from fuzzywuzzy import fuzz, process

#--------------
# Paths
#--------------
main_path = 'L:/.shortcut-targets-by-id/12-fuK40uOBz3FM-OXbZtk_OSlYcGpzpa/PovertyPredictionRealTime/data'
o1_path = os.path.join(main_path, '1_raw/peru/big_data/admin')
d1_path = os.path.join(main_path, '2_intermediate')

#%%





# ============================
#
# 1. LOADING DATA
#
# ============================

# %%

#--------------
# Loading
#--------------
file = os.path.join(o1_path, 'adulto_nutrition/DAT Adulto_Mayor_04_Nutricional.csv')
df = pd.read_csv(file, encoding='ISO-8859-1')

#%%





# ============================
#
# 2. CLEANING DATA
#
# ============================

# %%

#--------------
# Copy
#--------------
df2 = df.copy()

#--------------
# Cleaning column names
#--------------

# Dictionary of matching column names & new column names
name_mapping = {
    'anio'                : 'year',
    'mes'                 : 'month',
    'ubigeo'              : 'ubigeo',
    'grupo nutricional'   : 'test',
    'detalle nutricional' : 'result',
    'casos'               : 'cases'
}

# Map function to rename old matching names with new column names
def map_column_name(name):
    best_match, score = process.extractOne(name.lower(), name_mapping.keys())
    if score >= 80:
        return name_mapping[best_match]
    return name

# Applying the map function to rename variables based on the dictionary
df2 = df2.rename(columns=map_column_name)

#--------------
# Filtering columns
#--------------
df2 = df2[list(name_mapping.values())]

#--------------
# Cleaning character variables' values
#--------------

# Function that cleans utf-8 special characters and converts to lower cases
def clean_and_normalize(text):
    if isinstance(text, str):
        return unidecode(text).lower()
    else:
        return text

# Applying function
df2 = df2.map(clean_and_normalize)      

# Keeping important observations
keywords = ["indice de masa corporal", "mini valoracion nutricional"]
df2 = df2[df2['test'].apply(lambda x: any(fuzz.ratio(keyword, x) >= 90 for keyword in keywords))]

# %%





# ============================
#
# 3. AGGREGATING DATA
#
# ============================

# %%

#--------------
# Copy
#--------------
df3 = df2.copy()

#--------------
# Aggregating by result
#--------------
df3 = df3.groupby(['year', 'month', 'ubigeo', 'result']).agg({'cases': 'sum'}).reset_index()

#--------------
# Wide reshape (pivot = result)
#--------------
df3 = df3.pivot(index=['year', 'month', 'ubigeo'], columns='result', values=['cases'])
df3.columns = [f'{col[0]}_{col[1]}' for col in df3.columns]
df3 = df3.reset_index()

#--------------
# Replacing NaN with 0
#--------------
df3 = df3.fillna(0)

#--------------
# Cleaning column names
#--------------

# Dictionary of matching column names & new column names
name_mapping = {
    'cases_bien'          : 'nutri1',
    'cases_delgadez'      : 'nutri2',
    'cases_desnutrido'    : 'nutri3',
    'cases_normal'        : 'bmi1',
    'cases_sobrepeso'     : 'bmi2',
    'cases_obesidad'      : 'bmi3'
}

# Map function to rename old matching names with new column names
def map_column_name(name):
    best_match, score = process.extractOne(name.lower(), name_mapping.keys())
    if score >= 80:
        return name_mapping[best_match]
    return name

# Applying the map function to rename variables based on the dictionary
df3 = df3.rename(columns=map_column_name)

# %%





# ============================
#
# 4. CREATING VARIABLES
#
# ============================

# %%

#--------------
# Copy
#--------------
df4 = df3.copy()

#--------------
# Nutrition variables
#--------------
df4['nutri_tot'] = df4[['nutri1', 'nutri2', 'nutri3']].sum(axis=1)

nutri_cols = ['nutri1', 'nutri2', 'nutri3']
for col in nutri_cols:
    df4[f'{col}_per'] = df4[col] / df4['nutri_tot']

#--------------
# BMI variables
#--------------
df4['bmi_tot'] = df4[['bmi1', 'bmi2', 'bmi3']].sum(axis=1)

bmi_cols = ['bmi1', 'bmi2', 'bmi3']
for col in bmi_cols:
    df4[f'{col}_per'] = df4[col] / df4['bmi_tot']

#--------------
# Keeping variables
#--------------
keep_list = ['year', 'month', 'ubigeo', 
             'nutri1_per','nutri2_per', 'nutri3_per', 'nutri_tot',
             'bmi1_per', 'bmi2_per', 'bmi3_per', 'bmi_tot'
               ]
df4 = df4[keep_list]

#--------------
# Filling NaN
#--------------
df4 = df4.fillna(0)

# %%





# ============================
#
# 5. EXPORTING DATA
#
# ============================

# %%

#--------------
# Copying dataframe
#--------------
df5 = df4.copy()

#--------------
# Exporting final dataframe
#--------------
export_file = os.path.join(d1_path, 'senior_nutrition.csv')
df5.to_csv(export_file, index=False, encoding='utf-8')

# %%


