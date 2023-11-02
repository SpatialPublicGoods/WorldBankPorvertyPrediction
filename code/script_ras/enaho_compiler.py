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
path_o1 = os.path.join(main_path, '1_raw/peru/data/ENAHO')
path_d1 = os.path.join(main_path, '2_intermediate')

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
years_list = os.listdir(path_o1)[:-1]

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
    folder = os.path.join(path_o1, year)

    #--------------
    # Defining files to open
    #--------------
    hou_file = os.path.join(folder, f'enaho01-{year}-100.dta')
    sum_file = os.path.join(folder, f'sumaria-{year}.dta')

    #--------------
    # Opening files
    #--------------
    df_hou = pd.read_stata(hou_file, convert_categoricals=False)
    df_sum = pd.read_stata(sum_file, convert_categoricals=False)

    #--------------
    # Renaming variables
    #--------------
    if year == '2012': df_hou = df_hou.rename(columns={'ccpp': 'codccpp'})
    df_hou = df_hou.rename(columns={'aÑo' : 'year', 'mes' : 'month'})
    df_sum = df_sum.rename(columns={'aÑo' : 'year', 'mes' : 'month'})

    #--------------
    # Keeping variables
    #--------------
    keep_hou = ['year', 'month', 'conglome', 'vivienda', 'hogar', 'ubigeo', 'estrato', 'dominio',
                'result', 'factor07', 'codccpp', 'longitud', 'latitud']
    keep_sum = ['conglome', 'vivienda', 'hogar',
                'percepho', 'mieperho', 'totmieho', 'linpe', 'linea', 'pobreza', 'ingmo1hd', 'ingmo2hd', 'inghog1d', 'inghog2d', 'gashog1d', 'gashog2d']
    df_hou = df_hou.loc[:, keep_hou]
    df_sum = df_sum.loc[:, keep_sum]

    #--------------
    # Merging dataframes
    #--------------
    merged_df = pd.merge(df_hou, df_sum, on=['conglome', 'vivienda', 'hogar'], how='inner')

    #--------------
    # Appending merged dataframe
    #--------------
    dfs.append(merged_df)

#--------------
# Vertically concatenating all dataframes in the list
#--------------
df_enaho = pd.concat(dfs, ignore_index=True)

#--------------
# Creating expansion factors
#--------------
df_enaho['hh_factor']  = df_enaho['factor07']
df_enaho['pop_factor'] = df_enaho['factor07']*df_enaho['mieperho']

#--------------
# Exporting final dataframe
#--------------
export_file = os.path.join(path_d1, 'income_panel.csv')
df_enaho.to_csv(export_file, index=False, encoding='utf-8')

#%%









# Monthly per capita monetary income
df_enaho2['inc_pc'] = (df_enaho2['ingmo1hd'] / df_enaho2['mieperho'])/12
df_enaho2['log_inc_pc'] = np.log(df_enaho2['inc_pc']+1)




# ============================
#
# 2. SOME COOL GRAPHS
#
# ============================

# %%

#--------------
# Copying dataset
#--------------
df_enaho2 = df_enaho.copy()

#--------------
# Figures
#--------------

# Years column
years = df_enaho2['year'].unique()

# One histogram for each year
for year in years:
    data_year = df_enaho2[df_enaho2['year'] == year]
    hh_inc_pc= data_year['log_inc_pc']
    
    # Crea el histograma utilizando los ingresos de las familias y el número de bins deseado (por ejemplo, 20)
    plt.figure(figsize=(8, 6))
    plt.hist(hh_inc_pc, bins=200, density=True, alpha=0.7, color='b')
    plt.title(f'Monthly household percapita log-income, {year}')
    plt.xlabel('')
    plt.ylabel('Density')
    plt.xlim(0, 10)
    plt.show()

# %%

