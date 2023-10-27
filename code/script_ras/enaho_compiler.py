##############################################
#
# COMPILING ENAHO DATASETS
#
##############################################





# ============================
#
# 1. SETTINGS
#
# ============================

#%%

#--------------
# Libraries
#--------------
import os
import pandas as pd
from unidecode import unidecode

#--------------
# Paths
#--------------
main_path = 'L:/.shortcut-targets-by-id/12-fuK40uOBz3FM-OXbZtk_OSlYcGpzpa/PovertyPredictionRealTime/data'
path_o1 = os.path.join(main_path, '1_raw/peru/data/ENAHO')
path_d1 = os.path.join(main_path, '2_intermediate')

#%%










# ============================
#
# 2. ASSEMBLING DATA
#
# ============================

#%%

#--------------
# Years list
#--------------
years_list = os.listdir(path_o1)[:-1]

#--------------
# Final dataframe
#--------------
df_enaho = pd.DataFrame()

#--------------
# Assembly loop
#--------------
for i in range(0, len(years_list)):

    #--------------
    # Defining parameters
    #--------------
    year   = years_list[i]
    folder = os.path.join(path_o1, year)

    #--------------
    # Housing dataset
    #--------------
    df_hou = pd.read_stata(os.path.join(folder, 'enaho01-'+ year +'-100.dta'), convert_categoricals=False)

    # Renaming variable
    if year == '2012': df_hou = df_hou.rename(columns={'ccpp': 'codccpp'})

    #--------------
    # Sumaria dataset
    #--------------
    df_sum = pd.read_stata(os.path.join(folder, 'sumaria-'+ year +'.dta'), convert_categoricals=False)

    #--------------
    # Merging datasets
    #--------------
    merged_df = pd.merge(df_hou, df_sum, on=['conglome', 'vivienda', 'hogar'], how='inner')

    #--------------
    # Appending dataset
    #--------------
    df_enaho = pd.concat([df_enaho, merged_df], ignore_index=True)

#--------------
# Renaming
#--------------
df_enaho = df_enaho.rename(columns={'aÑo_x': 'year',
                                    'mes_x': 'month',
                                    'ubigeo_x':'ubigeo', 
                                    'dominio_x':'dominio',
                                    'estrato_x':'estrato',
                                    'factor07_x':'factor07'})

#--------------
# Keeping variables
#--------------
keep_vars = ['year', 'month', 'conglome', 'vivienda', 'hogar', 'ubigeo', 'estrato', 'result', 'factor07', 'codccpp', 'longitud', 'latitud',
            'percepho', 'mieperho', 'totmieho', 'linpe', 'linea', 'pobreza','ingmo1hd', 'ingmo2hd', 'inghog1d', 'inghog2d', 'gashog1d', 'gashog2d']
df_enaho = df_enaho.loc[:, keep_vars]

#%%





# %%
