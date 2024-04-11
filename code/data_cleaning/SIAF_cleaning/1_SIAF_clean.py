##############################################
#
# SCRIPT ADAPTED FROM HERNAN'S JUPYTER FILE
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

#--------------
# Paths
#--------------
main_path = 'L:/.shortcut-targets-by-id/12-fuK40uOBz3FM-OXbZtk_OSlYcGpzpa/PovertyPredictionRealTime/data'
o1_path = os.path.join(main_path, '1_raw/peru/big_data/admin')
d1_path = os.path.join(main_path, '2_intermediate')

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

# Ingreso data can not be loaded into python, we just work with Gasto data
# Ingreso data is cleaned in the STATA dofile 

gasto2020   = pd.read_csv(os.path.join(o1_path, 'SIAF/2020-Gasto.csv'),   encoding='iso-8859-1', on_bad_lines='skip')
gasto2021   = pd.read_csv(os.path.join(o1_path, 'SIAF/2021-Gasto.csv'),   encoding='iso-8859-1', on_bad_lines='skip')
#ingreso2020 = pd.read_csv(os.path.join(o1_path, 'SIAF/2020-Ingreso.csv'), encoding='iso-8859-1', on_bad_lines='skip')
#ingreso2021 = pd.read_csv(os.path.join(o1_path, 'SIAF/2021-Ingreso.csv'), encoding='iso-8859-1', on_bad_lines='skip')

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
gasto2020_c1 = gasto2020.copy()
gasto2021_c1 = gasto2021.copy()

#--------------
# Cleaning columns
#--------------
gasto2020_c1 = gasto2020_c1[['DEPARTAMENTO_EJECUTORA','DEPARTAMENTO_EJECUTORA_NOMBRE','PROVINCIA_EJECUTORA','PROVINCIA_EJECUTORA_NOMBRE','DISTRITO_EJECUTORA','DISTRITO_EJECUTORA_NOMBRE','ANO_EJE','MES_EJE','NIVEL_GOBIERNO','NIVEL_GOBIERNO_NOMBRE','FUNCION_NOMBRE','CATEGORIA_GASTO_NOMBRE','MONTO_PIA','MONTO_PIM','MONTO_COMPROMETIDO','MONTO_DEVENGADO','MONTO_GIRADO']]
gasto2021_c1 = gasto2021_c1[['DEPARTAMENTO_EJECUTORA','DEPARTAMENTO_EJECUTORA_NOMBRE','PROVINCIA_EJECUTORA','PROVINCIA_EJECUTORA_NOMBRE','DISTRITO_EJECUTORA','DISTRITO_EJECUTORA_NOMBRE','ANO_EJE','MES_EJE','NIVEL_GOBIERNO','NIVEL_GOBIERNO_NOMBRE','FUNCION_NOMBRE','CATEGORIA_GASTO_NOMBRE','MONTO_PIA','MONTO_PIM','MONTO_COMPROMETIDO','MONTO_DEVENGADO','MONTO_GIRADO']]

#--------------
# Lowercase columns
#--------------
gasto2020_c1.columns = gasto2020_c1.columns.str.lower()
gasto2021_c1.columns = gasto2021_c1.columns.str.lower()





# ============================
#
# 3. EXPORTING DATA
#
# ============================

# %%

#--------------
# Exporting final dataframe
#--------------
gasto2020_c1.to_csv(os.path.join(o1_path, 'SIAF/2020Gasto.csv'), sep=',', index=False, encoding='utf-8')
gasto2021_c1.to_csv(os.path.join(o1_path, 'SIAF/2021Gasto.csv'), sep=',', index=False, encoding='utf-8')

# %%