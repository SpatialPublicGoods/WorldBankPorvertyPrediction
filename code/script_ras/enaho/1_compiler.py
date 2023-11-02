##############################################
#
# COMPILING ENAHO DATASETS
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

#--------------
# Paths
#--------------
main_path = 'L:/.shortcut-targets-by-id/12-fuK40uOBz3FM-OXbZtk_OSlYcGpzpa/PovertyPredictionRealTime/data'
o1_path = os.path.join(main_path, '1_raw/peru/data/ENAHO')
d1_path = os.path.join(main_path, '2_intermediate')

#%%





# ============================
#
# 1. ASSEMBLING DATA
#
# ============================

#%%

#--------------
# Years list
#--------------
years_list = os.listdir(o1_path)[:-1]

#--------------
# Empty list to store dataframes
#--------------
dfs = []

#--------------
# Assembly loop
#--------------
for year in years_list:

    #--------------
    # Defining wich folder to use
    #--------------
    enaho_folder = os.path.join(o1_path, year)

    #--------------
    # Defining files to open
    #--------------
    hou_file = os.path.join(enaho_folder, f'enaho01-{year}-100.dta')
    sum_file = os.path.join(enaho_folder, f'sumaria-{year}.dta')

    #--------------
    # Opening files
    #--------------
    hou_df = pd.read_stata(hou_file, convert_categoricals=False)
    sum_df = pd.read_stata(sum_file, convert_categoricals=False)

    #--------------
    # Renaming variables
    #--------------
    if year == '2012': hou_df = hou_df.rename(columns={'ccpp': 'codccpp'})
    hou_df = hou_df.rename(columns={'aÑo' : 'year', 'mes' : 'month'})
    sum_df = sum_df.rename(columns={'aÑo' : 'year', 'mes' : 'month'})

    #--------------
    # Keeping variables
    #--------------
    hou_keep = ['year', 'month', 'conglome', 'vivienda', 'hogar', 'ubigeo', 'estrato', 'dominio',
                'result', 'factor07', 'codccpp', 'longitud', 'latitud']
    sum_keep = ['conglome', 'vivienda', 'hogar',
                'percepho', 'mieperho', 'totmieho', 'linpe', 'linea', 'pobreza', 'ingmo1hd', 'ingmo2hd', 'inghog1d', 'inghog2d', 'gashog1d', 'gashog2d']
    hou_df = hou_df.loc[:, hou_keep]
    sum_df = sum_df.loc[:, sum_keep]

    #--------------
    # Merging dataframes
    #--------------
    ids = ['conglome', 'vivienda', 'hogar']
    merged_df = pd.merge(hou_df, sum_df, on=ids, how='inner')

    #--------------
    # Appending merged dataframe
    #--------------
    dfs.append(merged_df)

#--------------
# Vertically concatenating all dataframes in the list
#--------------
enaho_df = pd.concat(dfs, ignore_index=True)

#--------------
# Creating expansion factors
#--------------
enaho_df['hh_factor']  = enaho_df['factor07']
enaho_df['pop_factor'] = enaho_df['factor07']*enaho_df['mieperho']

# %%





# ============================
#
# 2. EXPORTING DATA
#
# ============================

# %%

#--------------
# Copying dataframe
#--------------
enaho_df2 = enaho_df.copy()

#--------------
# Exporting final dataframe
#--------------
export_file = os.path.join(d1_path, 'income_panel.csv')
enaho_df2.to_csv(export_file, index=False, encoding='utf-8')

#%%