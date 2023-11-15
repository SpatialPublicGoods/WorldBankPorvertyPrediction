##############################################
#
# INCOME FIGURES
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
path_o1 = os.path.join(main_path, '2_intermediate')
path_d1 = os.path.join(main_path, '2_intermediate') # ?????????????????

#%%





# ============================
#
# 1. LOADING INCOME DATA
#
# ============================

# %%

#--------------
# Loading
#--------------
income_file = os.path.join(path_o1, 'income_panel.csv')
income_df = pd.read_csv(income_file)

#%%





# ============================
#
# 2. SOME COOL GRAPHS
#
# ============================

# %%

#--------------
# Copy
#--------------
income_df2 = income_df.copy()

#--------------
# Monthly per capita monetary income
#--------------
income_df2['inc_pc'] = (income_df2['ingmo1hd'] / income_df2['mieperho'])/12
income_df2['log_inc_pc'] = np.log(income_df2['inc_pc']+1)

#--------------
# Figures
#--------------

# Years column
years = income_df2['year'].unique()

# One histogram for each year
for year in years:
    data_year = income_df2[income_df2['year'] == year]
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