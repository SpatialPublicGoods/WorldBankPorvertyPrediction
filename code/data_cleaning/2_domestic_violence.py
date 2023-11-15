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
y0 = 2016
y1 = 2019

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
    df   = pd.read_csv(file, encoding='iso-8859-1', on_bad_lines='skip')

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

    ### If date is string
    if df['date'].dtype == 'object':
        date_parts = df['date'].str.split('/', expand=True)
        df['year'] = pd.to_numeric(date_parts[2], errors='coerce').fillna(0).astype(int)
        df['month'] = pd.to_numeric(date_parts[0], errors='coerce').fillna(0).astype(int)
        df['day'] = pd.to_numeric(date_parts[1], errors='coerce').fillna(0).astype(int)

    ### If date is numeric
    elif pd.api.types.is_numeric_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], unit='D', origin='1899-12-30')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day

    ### Creating quartes variable:
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
final_df = final_df[final_df['year']>=2016]

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
# Aggregating by year, month
#--------------
df_aggregated = df.groupby(['year', 'month']).agg({'cases_tot': 'sum'}).reset_index()
df_aggregated['date'] = pd.to_datetime(df_aggregated[['year', 'month']].assign(DAY=1))
df_aggregated = df_aggregated.sort_values(by='date')

df_aggregated

# %%

#--------------
# Graphing
#--------------

# Graph size
plt.figure(figsize=(15, 8))

# Graph
plt.bar(df_aggregated['date'], df_aggregated['cases_tot'], width=20)

# Title and axis labels
plt.title('Evolution of domestic violence, by month, 2016 - 2019')
plt.xlabel('Date')
plt.ylabel('Total reported cases')

# x-axis dates configuration
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)

# Show graph
plt.show()

# %%
