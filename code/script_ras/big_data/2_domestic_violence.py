##############################################
#
# WORKING DOMESTIC VIOLENCE DATA
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

#--------------
# Parameters
#--------------
freq = 'm'
y0 = 2016
y1 = 2018

#%%





# ============================
#
# 1. CLEANING FUNCTION
#
# ============================

# %%

def data_cleaning(year_to_clean, chosen_frequency):

    #--------------
    # Opening
    #--------------
    file = os.path.join(o1_path, 'violencia familiar/SIDPOL_' + str(year_to_clean) + '_Violencia_familiar.csv')
    df   = df = pd.read_csv(file, encoding='UTF-8')

    #--------------
    # Cleaning column names
    #--------------

    # Dictionary of matching column names & new column names
    name_mapping = {
        'fecha'        : 'date',
        'ubigeo hecho' : 'ubigeo',
        'ubigeo cia'   : 'id'
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
    # Creating date variables
    #--------------

    df.dropna(inplace=True)

    date_parts = df['date'].str.split('/', expand=True)
    df['year']  = pd.to_numeric(date_parts[2], errors='coerce').fillna(0).astype(int)
    df['month'] = pd.to_numeric(date_parts[0], errors='coerce').fillna(0).astype(int)
    df['day']   = pd.to_numeric(date_parts[1], errors='coerce').fillna(0).astype(int)

    df.loc[(df['month'] <= 3)                     , 'quarter'] = 1
    df.loc[(df['month'] >= 4) & (df['month'] <= 6), 'quarter'] = 2
    df.loc[(df['month'] >= 7) & (df['month'] <= 9), 'quarter'] = 3
    df.loc[(df['month'] >= 10)                    , 'quarter'] = 4
    df['quarter'] = df['quarter'].astype(int)

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
    agg_list = agg_map.get(chosen_frequency, [])

    # Aggregating by aggregation list
    df = df.groupby(agg_list).agg({'id': 'count'}).reset_index()

    #--------------
    # Sorting data
    #--------------
    df = df.sort_values(by=agg_list)

    #--------------
    # Renaming
    #--------------
    df = df.rename(columns={'id': 'cases_tot'})

    #--------------
    # Returning clean data
    #--------------
    return df

# %%





# ============================
#
# 2. APPLYING DATA CLEANING
#
# ============================

# %%

#--------------
# Final data
#--------------
append_df = []

#--------------
# Cleaning loop
#--------------
for i in range(y0, y1 + 1):
    df_i = data_cleaning(i, freq)
    append_df.append(df_i)
    print(str(i) + ' done')

#--------------
# Appending
#--------------
final_df = pd.concat(append_df, ignore_index=True)

#--------------
# Cleaning
#--------------
final_df['ubigeo']  = final_df['ubigeo'].astype(int)
final_df = final_df.loc[final_df['year'] >= 2016]

# %%





# ============================
#
# 3. EXPORTING DATA
#
# ============================

# %%

#--------------
# Exporting final dataframe
#--------------
export_file = os.path.join(d1_path, 'domestic_violence.csv')
final_df.to_csv(export_file, index=False, encoding='utf-8')

# %%





# ============================
#
# 4. COOL GRAPHS
#
# ============================

# %%

#--------------
# Copy
#--------------
df = final_df.copy()


#--------------
# Aggregating by month
#--------------
agg_list = ['year', 'month']
df = df.groupby(agg_list).agg({'cases_tot': 'count'}).reset_index()


df

# %%





# Para cada año, filtrar los datos y hacer un histograma
for year in sorted(years):
    # Filtra el DataFrame por el año actual en el bucle
    df_year = df[df['year'] == year]
    
    # Agrupa los datos por 'month' y suma los 'cases'
    monthly_cases = df_year.groupby('year', 'month')['cases_tot'].sum()
    
    # Crea un histograma con la suma de casos por mes
    plt.figure(figsize=(10, 6))
    plt.bar(monthly_cases.index, monthly_cases.values)
    
    # Establece el título y las etiquetas
    plt.title(f'Total cases of domestic violence per month, {year}')
    plt.xlabel('Month')
    plt.ylabel('Total cases')
    plt.xticks(monthly_cases.index, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # Muestra el gráfico
    plt.show()



# %%