##############################################
#
# WORKING MAX TEMPERATURE DATA
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
file    = os.path.join(o1_path, 'Tmax.csv')
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
# 3. GRAPHS
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

monthly_yearly_stats[monthly_yearly_stats['missing_count'] != 0]

# %%










