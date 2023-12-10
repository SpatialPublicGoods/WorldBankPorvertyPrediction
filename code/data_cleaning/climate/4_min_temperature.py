##############################################
#
# WORKING MIN TEMPERATURE DATA
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
o1_path = os.path.join(main_path, '1_raw/peru/big_data/other/clima')
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
file    = os.path.join(o1_path, 'Tmin.csv')
data_df = pd.read_csv(file, encoding='iso-8859-1', on_bad_lines='skip')

# %%





# ============================
#
# 2. MERGING DATA
#
# ============================

# %%

#--------------
# Copy
#--------------
df = data_df.copy()

#--------------
# Creating variables
#--------------

# year
df['year'] = ((df['Month'] - 1) // 12) + 2013

# month
df['month'] = df['Month'] % 12
df.loc[df['month'] == 0, 'month'] = 12

#--------------
# Droping
#--------------
df = df.drop('Month', axis=1)

#--------------
# Renaming
#--------------
df.rename(columns={'Conglomerado ID': 'conglome'}, inplace=True)

#--------------
# Merging
#--------------
file_m = os.path.join(d1_path, 'clusters.csv')
df_m   = pd.read_csv(file_m, encoding='iso-8859-1', on_bad_lines='skip')
df2 = pd.merge(df_m, df, on=['conglome', 'year', 'month'], how='left', indicator=True)

# %%





# ============================
#
# 3. MISSING STATS
#
# ============================

# %%

#--------------
# Copy
#--------------
df3 = df2.copy()

#--------------
# Cleaning
#--------------
monthly_yearly_stats = df3.groupby(['year', 'month']).agg({'_merge': lambda x: (x == 'left_only').sum()})
monthly_yearly_stats.rename(columns={'_merge': 'missing_count'}, inplace=True)
monthly_yearly_stats['total_count'] = df3.groupby(['year', 'month'])['_merge'].count()
monthly_yearly_stats['missing_percentage'] = (monthly_yearly_stats['missing_count'] / monthly_yearly_stats['total_count']) * 100

#--------------
# Missings
#--------------
monthly_yearly_stats[monthly_yearly_stats['missing_count'] != 0]

# %%





# ============================
#
# 4. GRAPHS: HISTOGRAMS
#
# ============================

# %%

#--------------
# Copy
#--------------
df3 = df2.copy()

#--------------
# Graph
#--------------

# Obtener los años únicos
unique_years = df['year'].unique()
unique_years.sort()  # Asegúrate de que los años estén en orden

# Crear un histograma para cada año
for year in unique_years:
    # Filtrar el DataFrame para el año actual
    df_year = df[df['year'] == year]
    
    # Crear un histograma para la columna 'Mean' del año actual
    plt.figure(figsize=(10, 6))  # Tamaño de la figura
    plt.hist(df_year['Mean'], bins=200, color='blue', alpha=0.7)  # Ajusta los bins como sea necesario
    plt.title(f'Monthly "Mean" Min Temperature Distribution: {year}')
    plt.xlabel('Mean')
    plt.ylabel('Frequency')
    plt.xlim(-10, 40)
    plt.show()





# ============================
#
# 5. GRAPHS: EMPIRICAL CDF
#
# ============================

# %%

#--------------
# Copy
#--------------
df3 = df2.copy()

#--------------
# Graph
#--------------

# log(Mean + 1)
df3['log_Mean'] = np.log1p(df3['Mean'])

# Seaborn configuration
sns.set(style='whitegrid')

# unique years
unique_years = df3['year'].unique()
unique_years.sort()

# Combined CDF graph figure
plt.figure(figsize=(12, 8))

# Creating CDF
for year in unique_years:
    sns.ecdfplot(data=df3[df3['year'] == year], x='log_Mean', label=str(year))

# Title and axis labels
plt.title('Empirical CDF of Average log(Mean+1) Min Temperature, per year', fontsize=16)
plt.xlabel('log(Mean+1)', fontsize=14)
plt.ylabel('ECDF', fontsize=14)

plt.ylim(0, 1.02)
plt.xlim(0, 4)

plt.legend(title='Year', title_fontsize='13', fontsize='12')

plt.show()

# %%
















# %%























