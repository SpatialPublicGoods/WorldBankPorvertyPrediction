##############################################
#
# WORKING PUBLIC EXPENDITURE DATA
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
file_name = os.path.join(o1_path, 'Gasto_allyears_ubigeos/Gasto_allyears_ubigeos.csv')
df1       = pd.read_csv(file_name, encoding='iso-8859-1', on_bad_lines='skip')

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
df2 = df1.copy()

#--------------
# Cleaning column names
#--------------

### Dictionary of matching column names & new column names
name_mapping = {
    'ubigeo_inei'           : 'ubigeo',
    'ano_eje'               : 'year',
    'mes_eje'               : 'month',
    'monto_pia'             : 'pia',
    'monto_devengado'       : 'dev',
    'nivel_gobierno_nombre' : 'institution',
    'funcion_nombre'        : 'type'
}

### Map function to rename old matching names with new column names
def map_column_name(name):
    best_match, score = process.extractOne(name.lower(), name_mapping.keys())
    if score >= 99:
        return name_mapping[best_match]
    return name

### Applying the map function to rename variables based on the dictionary
df2 = df2.rename(columns=map_column_name)

#--------------
# Filtering columns
#--------------
df2 = df2[list(name_mapping.values())]

#--------------
# Filtering rows by local institutions
#--------------
df2 = df2[df2['institution'] == 'GOBIERNOS LOCALES'].reset_index(drop=True)
df2 = df2.drop(columns=['institution'])










# ============================
#
# 3. CREATING VARIABLES
#
# ============================

# %%

#--------------
# Copy
#--------------
df3 = df2.copy()

#--------------
# Quarter (date)
#--------------
df3.loc[(df3['month'] <= 3)                      , 'quarter'] = 1
df3.loc[(df3['month'] >= 4) & (df3['month'] <= 6), 'quarter'] = 2
df3.loc[(df3['month'] >= 7) & (df3['month'] <= 9), 'quarter'] = 3
df3.loc[(df3['month'] >= 10)                     , 'quarter'] = 4
df3['quarter'] = df3['quarter'].astype(int)

#--------------
# Changing 'type' values
#--------------
new_values = {
    ' '                             : 'na',
    'TRANSPORTE'                    : 'transport',
    'SALUD'                         : 'health',
    'PREVISION SOCIAL'              : 'social_security',
    'AMBIENTE'                      : 'environment',
    'ORDEN PUBLICO Y SEGURIDAD'     : 'public_order',
    'SANEAMIENTO'                   : 'sanitation',
    'EDUCACION'                     : 'education',
    'AGROPECUARIA'                  : 'agriculture',
    'CULTURA Y DEPORTE'             : 'culture_sports',
    'COMERCIO'                      : 'commerce',
    'PLANEAMIENTO, GESTION Y RESERVA DE CONTINGENCIA': 'planning_management',
    'DEUDA PUBLICA'                 : 'public_debt',
    'PROTECCION SOCIAL'             : 'social_protection',
    'VIVIENDA Y DESARROLLO URBANO'  : 'urban_development',
    'ENERGIA'                       : 'energy',
    'MEDIO AMBIENTE'                : 'environment',
    'TURISMO'                       : 'tourism',
    'TRABAJO'                       : 'labor',
    'COMUNICACIONES'                : 'communications',
    'JUSTICIA'                      : 'justice',
    'PESCA'                         : 'fishing',
    'INDUSTRIA'                     : 'industry',
    'MINERIA'                       : 'mining'
}

df3['type'] = df3['type'].replace(new_values).str.lower()

#--------------
# Reshaping variables
#--------------

### Map for aggregation list
agg_map = {
    'd': ['ubigeo', 'year', 'quarter', 'month', 'day'],
    'm': ['ubigeo', 'year', 'quarter', 'month'],
    'q': ['ubigeo', 'year', 'quarter'],
    'y': ['ubigeo', 'year']
}

### Defining aggregation list
agg_list = agg_map.get(freq, [])

### Aggregating
df3 = df3.groupby(agg_list + ['type']).agg(
    sum_pia=('pia', 'sum'),
    sum_dev=('dev', 'sum')
).reset_index()

### Renaming 'sum_pia' and 'sum_dev'
df3 = df3.rename(columns={
    'sum_pia': 'pubexp_pia',
    'sum_dev': 'pubexp_dev'
})

### Reshaping by 'ubigeo', 'year', 'month'
df3 = df3.pivot_table(
    index=agg_list,
    columns='type',
    values=['pubexp_pia', 'pubexp_dev'],
    aggfunc='sum'
).reset_index()

# Flattening aggregation index
df3.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df3.columns.values]

#--------------
# Sorting data
#--------------
df3 = df3.sort_values(by=agg_list)

# %%










# ============================
#
# 4. EXPORTING DATA
#
# ============================

# %%

#--------------
# Copy
#--------------
final_df = df3.copy()

#--------------
# Exporting final dataframe
#--------------
export_file = os.path.join(d1_path, 'public_expenditure.csv')
final_df.to_csv(export_file, index=False, encoding='utf-8')

# %%