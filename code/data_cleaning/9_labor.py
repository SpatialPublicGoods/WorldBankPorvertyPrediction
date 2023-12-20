##############################################
#
# WORKING LABOR DATA
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
from modules.utils_general import utils_general

#--------------
# Paths
#--------------
main_path = 'L:/.shortcut-targets-by-id/12-fuK40uOBz3FM-OXbZtk_OSlYcGpzpa/PovertyPredictionRealTime/data'
o1_path = os.path.join(main_path, '1_raw/peru/big_data/admin')
d1_path = os.path.join(main_path, '2_intermediate')

#--------------
# Parameters
#--------------
freq = 'y'
#%%





# ============================
#
# 1. OPENING DATA
#
# ============================

# %%

#--------------
# Parameters
#--------------
files_sheets = {
    2014: ('indicadores_2014 (1).xls',                      'DISTRITOS_EMPRESAS', 'TRAB_SEXO',     'TRAB_CAT_OCUP_XX',  'REM_Sit_Ed'),
    2015: ('indicadores_2015 (1).xls',                      'EMPRESAS',           'Trab_Sexo',     'Trab_Cat_Ocup',     'Rem_Sit_Edu'),
    2016: ('indicadores_planilla_distrital (1).xls',        'EMPRESAS',           'Trab_mes_Sexo', 'Trab_Cat_Ocup_Mes', 'Rem_Sit_Edu'),
    2017: ('IND (1).xlsx',                                  'EMPRESAS_17',        'Trab_Sex_17',   'Trab_Cat_Ocup_17',  'Rem_Sit_Edu_17'),
    2018: ('DISTRITAL__2018 (1).xlsx',                      'EMPRESAS_18',        'Trab_Sex_18',   'Trab_Ocup_18',      'Rem_Sit_edu_18'),
    2019: ('DISTRITAL__2019_PLANILLA_ELECTRÓNICA (1).xlsx', 'EMPRESAS_19',        'Trab_Sex_19',   'Trab_Ocup_19',      'Rem_Sit_edu_19')
}

#--------------
# Dictionaries
#--------------
companies = {}
workers_sex = {}
workers_occupation = {}
salaries_education = {}

#--------------
# Opening files
#--------------
for year, (filename, company_sheet, worker_sex_sheet, worker_occupation_sheet, salary_sheet) in files_sheets.items():
    full_path = os.path.join(o1_path, 'planilla_electronica', filename)
    companies[year] = pd.read_excel(full_path, sheet_name=company_sheet)
    workers_sex[year] = pd.read_excel(full_path, sheet_name=worker_sex_sheet)
    workers_occupation[year] = pd.read_excel(full_path, sheet_name=worker_occupation_sheet)
    salaries_education[year] = pd.read_excel(full_path, sheet_name=salary_sheet)

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
companies_copies = {year: df.copy() for year, df in companies.items()}
workers_sex_copies = {year: df.copy() for year, df in workers_sex.items()}
workers_occupation_copies = {year: df.copy() for year, df in workers_occupation.items()}
salaries_education_copies = {year: df.copy() for year, df in salaries_education.items()}

#--------------
# Companies
#--------------
def process_dataframe(df, year):

    if year == 2019:
        df_processed = df.iloc[7:-4, 0:14]
    else:
        df_processed = df.iloc[6:-5]

    df_processed = df_processed.drop(df_processed.columns[1], axis=1)
    df_processed.columns = ['ubigeo'] + [str(i) for i in range(1, len(df_processed.columns))]
    df_processed = pd.melt(df_processed, id_vars=['ubigeo'], var_name='month', value_name='companies')  
    df_processed['month'] = df_processed['month'].astype(int)
    df_processed['year'] = year
    df_processed.reset_index(drop=True, inplace=True)

    return df_processed[['ubigeo', 'year', 'month', 'companies']]

for year in range(2014, 2020):
    companies_copies[year] = process_dataframe(companies_copies[year], year)

all_companies = pd.concat(companies_copies.values(), ignore_index=True)
all_companies = all_companies[(all_companies['ubigeo'] != '00000') & (all_companies['ubigeo'] != '000000')]
all_companies.sort_values(by=['ubigeo', 'year', 'month'], inplace=True)
all_companies.reset_index(drop=True, inplace=True)

#--------------
# Workers by sex
#--------------
def process_data_for_year(df, year):
    df = df.iloc[7:-5].drop(df.columns[1], axis=1)
    
    num_cols = (len(df.columns) - 1) // 3
    column_names = ['ubigeo'] + [f'{x}_{i}' for i in range(1, num_cols + 1) for x in ['si', 'm', 'f']]
    df.columns = column_names
    df = df.melt(id_vars='ubigeo', var_name='variable', value_name='value')
    df['month'] = df['variable'].str.extract(r'_(\d+)')[0].astype(int)
    df['type'] = df['variable'].str.extract(r'([smf]i?)')[0]
    df = df.pivot_table(index=['ubigeo', 'month'], columns='type', values='value', aggfunc='first').reset_index()
    df.columns.name = None
    df.rename(columns={'f': 'f', 'm': 'm', 'si': 'si'}, inplace=True)
    df['year'] = year

    return df

for year in range(2014, 2020):
    workers_sex_copies[year] = process_data_for_year(workers_sex_copies[year], year)

all_workers_sex = pd.concat(workers_sex_copies.values(), ignore_index=True)
all_workers_sex = all_workers_sex[(all_workers_sex['ubigeo'] != '00000') & (all_workers_sex['ubigeo'] != '000000')]
all_workers_sex.sort_values(by=['ubigeo', 'year', 'month'], inplace=True)
all_workers_sex.reset_index(drop=True, inplace=True)

#--------------
# Workers by ocupation type
#--------------

row_trim_values = {
    2014: 7,
    2015: 7,
    2016: 10,
    2017: 11,
    2018: 13,
    2019: 11
}

for year in range(2014, 2020):

    row_trim = row_trim_values.get(year)
    df = workers_occupation_copies[year].iloc[row_trim:-5].drop(workers_occupation_copies[year].columns[1], axis=1)
    
    #_________________________________
    if (year>=2014) & (year<=2016):
    #_________________________________
        df.columns = ['ubigeo'] + [f'{x}_{i}' for i in range(1, (len(df.columns) // 4) + 1) for x in ['exe', 'wor', 'emp', 'nd']]
        df = pd.melt(df, id_vars=['ubigeo'], var_name='variable', value_name='value')
        df['month'] = df['variable'].str.extract(r'_(\d+)$').astype(int)
        df['variable'] = df['variable'].str.extract(r'^(exe|wor|emp|nd)')
        df_final = df.pivot_table(index=['ubigeo', 'month'], columns='variable', values='value', aggfunc='first').reset_index()
        df_final.columns = ['ubigeo', 'month'] + sorted(df['variable'].unique().tolist())
        df_final['year'] = year
        
        workers_occupation_copies[year] = df_final[['ubigeo', 'year', 'month', 'exe', 'wor', 'emp', 'nd']]

    #_________________________________
    elif (year>=2017) & (year<=2019):
    #_________________________________
        new_column_names = ['ubigeo']
        categories = ['exe', 'wor', 'emp', 'nd']
        genders = ['si', 'm', 'f']
        for i in range(1, 13):
            for category in categories:
                for gender in genders:
                    new_column_names.append(f'{category}_{gender}_{i}')
        df.columns = new_column_names

        df_summed = df[['ubigeo']].copy()
        categories = ['exe', 'wor', 'emp', 'nd']
        for category in categories:
            for i in range(1, 13):
                cols_to_sum = [f'{category}_{gender}_{i}' for gender in ['si', 'm', 'f']]
                df_summed[f'{category}_{i}'] = df[cols_to_sum].sum(axis=1)

        df_melted = pd.melt(df, id_vars=['ubigeo'], var_name='variable', value_name='value')
        df_melted['month'] = df_melted['variable'].str.extract(r'_(\d+)$').astype(int)
        df_melted['category'] = df_melted['variable'].str.extract(r'^(exe|wor|emp|nd)')
        df_melted.drop('variable', axis=1, inplace=True)
        df_pivot = df_melted.pivot_table(index=['ubigeo', 'month'], columns='category', values='value', aggfunc='sum').reset_index()
        df_pivot = df_pivot[['ubigeo', 'month', 'exe', 'wor', 'emp', 'nd']]
        df_pivot.columns.name = None
        df_pivot['year'] = year
        workers_occupation_copies[year] = df_pivot[['ubigeo', 'year', 'month', 'exe', 'wor', 'emp', 'nd']]

all_workers_occupation = pd.concat(workers_occupation_copies, ignore_index=True)
all_workers_occupation = all_workers_occupation[(all_workers_occupation['ubigeo'] != '00000') & (all_workers_occupation['ubigeo'] != '000000')]
all_workers_occupation.sort_values(by=['ubigeo', 'year', 'month'], inplace=True)
all_workers_occupation.reset_index(drop=True, inplace=True)

#--------------
# Salaries by education
#--------------

row_trim_values = {
    2014: 7,
    2015: 6,
    2016: 7,
    2017: 10,
    2018: 11,
    2019: 11
}

col_trim_values = {
    2014: [0, 24],
    2015: [0, 24],
    2016: [0, 24],
    2017: [0, 68, 69, 70],
    2018: [0, 68, 69, 70],
    2019: [0, 68, 69, 70]
}

df_workers = all_workers_sex.copy()
df_workers['total'] = df_workers['f'] + df_workers['m'] + df_workers['si']
df_workers['total'].replace(0, np.nan, inplace=True)
df_workers['weight_f']  = (df_workers['f']  / df_workers['total']).fillna(0)
df_workers['weight_m']  = (df_workers['m']  / df_workers['total']).fillna(0)
df_workers['weight_si'] = (df_workers['si'] / df_workers['total']).fillna(0)
df_workers = df_workers[df_workers['month'] == 12][['ubigeo', 'year', 'weight_f', 'weight_m', 'weight_si']]

for year in range(2014, 2020):

    row_trim = row_trim_values.get(year)
    col_trim = col_trim_values.get(year)
    df = salaries_education_copies[year].iloc[row_trim:-5, col_trim]
    df['year'] = year
    
    #_________________________________
    if (year>=2014) & (year<=2016):
    #_________________________________    
        df.columns = ['ubigeo', 'salaries_mean', 'year']
        df.replace(0, np.nan, inplace=True)
        df['ubigeo'] = df['ubigeo'].astype(str)
        df = df.sort_values(by='ubigeo')[['ubigeo', 'year'] + [col for col in df.columns if col not in ['ubigeo', 'year']]]
        salaries_education_copies[year] = df

    #_________________________________
    elif (year>=2017) & (year<=2019):
    #_________________________________
        df.columns = ['ubigeo', 'salaries_si', 'salaries_m', 'salaries_f', 'year']
        df.replace(0, np.nan, inplace=True)
        df['ubigeo'] = df['ubigeo'].astype(str)
        df = df.sort_values(by='ubigeo')[['ubigeo', 'year'] + [col for col in df.columns if col not in ['ubigeo', 'year']]]

        df2 = pd.merge(df, df_workers, on=['ubigeo', 'year'], how='left')

        df2['salaries_mean'] = df2.apply(
        lambda row: np.nansum([
            row['weight_si'] * row['salaries_si'],
            row['weight_f'] * row['salaries_f'],
            row['weight_m'] * row['salaries_m']
        ]),
        axis=1
        )

        df2['salaries_mean'].replace(0, np.nan, inplace=True)
        salaries_education_copies[year] = df2.iloc[:, [0,1,-1]]

all_salaries = pd.concat(salaries_education_copies, ignore_index=True)
all_salaries = all_salaries[(all_salaries['ubigeo'] != '00000') & (all_salaries['ubigeo'] != '000000')]
all_salaries.sort_values(by=['ubigeo', 'year'], inplace=True)
all_salaries.reset_index(drop=True, inplace=True)

# %%





# ============================
#
# 3. MERGING DATA
#
# ============================

# %%

#--------------
# Cleaning ubigeos
#--------------
all_companies['ubigeo'] = all_companies['ubigeo'].astype(int)
all_workers_sex['ubigeo'] = all_workers_sex['ubigeo'].astype(int)
all_workers_occupation['ubigeo'] = all_workers_occupation['ubigeo'].astype(int)
all_salaries['ubigeo'] = all_salaries['ubigeo'].astype(int)

#--------------
# Merging
#--------------
m1_df = pd.merge(all_companies, all_workers_sex,        on=['ubigeo', 'year', 'month'], how='outer')
m2_df = pd.merge(m1_df        , all_workers_occupation, on=['ubigeo', 'year', 'month'], how='outer')
m3_df = pd.merge(m2_df        , all_salaries          , on=['ubigeo', 'year']         , how='outer')

#--------------
# Sorting
#--------------
m3_df.sort_values(by=['ubigeo', 'year', 'month'], inplace=True)

# %%





# ============================
#
# 4. FINAL CLEANING
#
# ============================

# %%

#--------------
# Copy
#--------------
df = m3_df.copy()

#--------------
# Creating variables
#--------------
df['workers_tot'] = df['f'] + df['m'] + df['si']
df['workers_tot'].replace(0, np.nan, inplace=True)

for col in ['f', 'm', 'si', 'exe', 'wor', 'emp', 'nd']:
    df[col] = (df[col] / df['workers_tot']).fillna(0).round(4)

df['workers_tot'] = df['workers_tot'].fillna(0).astype(int)

df.rename(columns={
    'f':   'workers_sex_f_perc',
    'm':   'workers_sex_m_perc',
    'si':  'workers_sex_si_perc',
    'exe': 'workers_type_exe_perc',
    'wor': 'workers_type_wor_perc',
    'emp': 'workers_type_emp_perc',
    'nd':  'workers_type_nd_perc'
}, inplace=True)

#--------------
# Cleaning
#--------------
df['companies'] = df['companies'].fillna(0).astype(int)

#--------------
# Sorting and ordering
#--------------
df = df.sort_values(by=['ubigeo', 'year', 'month'])

new_order = [
    'ubigeo', 'year', 'month', 'companies', 
    'workers_tot', 'workers_sex_f_perc', 'workers_sex_m_perc', 
    'workers_sex_si_perc', 'workers_type_exe_perc', 'workers_type_wor_perc', 
    'workers_type_emp_perc', 'workers_type_nd_perc', 'salaries_mean'
]

df = df[new_order]

# %%





# ============================
#
# 5. EXPORTING DATA
#
# ============================

# %%

#--------------
# Copy
#--------------
final_df = df.copy()

#--------------
# Exporting final dataframe
#--------------
export_file = os.path.join(d1_path, 'labor.csv')
final_df.to_csv(export_file, index=False, encoding='utf-8')

# %%