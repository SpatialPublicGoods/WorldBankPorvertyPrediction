##############################################
#
# ASSEMBLING AN INCOME POOL DATASET
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
path_d1 = os.path.join(main_path, '2_intermediate')

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
# 2. AGGREGATING DATA
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
# Aggregating at the cluster level
#--------------
pool_df = income_df2.groupby(['conglome', 'year', 'month']).agg({'inc_pc'      : 'mean',
                                                                 'hogar'       : 'count',
                                                                 'hh_factor'   : 'first',
                                                                 'mieperho'    : 'sum'
                                                                 }).reset_index()


pool_df

# %%



pool_df1 = income_df2.groupby(['conglome', 'year', 'month']).agg({'inc_pc'      : 'mean',
                                                                 'hogar'       : 'count',
                                                                 'hh_factor'   : 'mean',
                                                                 'mieperho'    : 'sum'
                                                                 }).reset_index()




pool_df[pool_df['hh_factor'] != pool_df1['hh_factor']]


# %%


pool_df1['hh_factor']



# %%
