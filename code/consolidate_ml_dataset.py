# Libraries
#--------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

#--------------
# Paths
#--------------

dataPath = 'J:/My Drive/PovertyPredictionRealTime/data'

admin_data_path = os.path.join(dataPath, '1_raw/peru/big_data/admin')

working = '3_working'

clean = '4_clean'

freq = 'm'

date = datetime.today().strftime('%Y-%m-%d')

#--------------

def read_enaho_panel():

    # Read csv
    enaho = pd.read_csv(os.path.join(dataPath, working, 'income.csv'), index_col=0, parse_dates=True).reset_index()

    enaho['ubigeo'] = 'U-' + enaho['ubigeo'].astype(str).str.zfill(6)

    enaho['year'] = enaho['year'].dt.year

    # Get sum of income and individuals at the conglome:
    enaho_conglome = enaho.groupby(['conglome', 'year']).sum().reset_index()

    # Compute income per capita:
    enaho_conglome['income_pc'] = enaho_conglome['ingmo1hd']/enaho_conglome['mieperho']
    
    # Compute gasto per capita:
    enaho_conglome['spend_pc'] = enaho_conglome['gashog1d']/enaho_conglome['mieperho']

    # Get lagged income_pc
    enaho_conglome['income_pc_lagged'] = enaho_conglome.groupby('conglome')['income_pc'].shift(1)
    enaho_conglome['spend_pc_lagged'] = enaho_conglome.groupby('conglome')['spend_pc'].shift(1)

    enaho_conglome = (enaho_conglome.sort_values(by=['conglome', 'year'])
                    .loc[:, ['conglome','year','income_pc','income_pc_lagged','spend_pc','spend_pc_lagged']]
                    )
    
    # Get conglome data to enaho:
    enaho = enaho.merge(enaho_conglome, on=['conglome', 'year'], how='left')

    return enaho


def read_domestic_violence_cases():

    domestic_violence = pd.read_csv(os.path.join(dataPath, working, 'domestic_violence.csv'), index_col=0).reset_index()

    domestic_violence['ubigeo'] = 'U-' + domestic_violence['ubigeo'].astype(str).str.zfill(6)

    domestic_violence = domestic_violence.groupby(['ubigeo','year', 'month']).first().reset_index()

    return domestic_violence


def read_police_reports():

    police_reports = pd.read_csv(os.path.join(dataPath, working, 'delitos_distrito_comisaria_clean_panel.csv'), index_col=0).reset_index()

    police_reports['ubigeo'] = 'U-' + police_reports['ubigeo'].astype(str).str.zfill(6)

    police_reports['year'] = police_reports['Year']

    # Group by granularity and frequency:
    police_reports_by_ubigeo = police_reports.groupby(['year', 'ubigeo']).sum().reset_index()

    return police_reports_by_ubigeo



def read_cargo_vehicles():

    cargo_vehicles = pd.read_csv(os.path.join(dataPath, working, 'cargo_vehicles.csv'), index_col=0).reset_index()

    cargo_vehicles['ubigeo'] = 'U-' + cargo_vehicles['ubigeo'].astype(str).str.zfill(6)
    

    return cargo_vehicles


# Load data:

enaho = read_enaho_panel()

domestic_violence = read_domestic_violence_cases()

police_reports_by_ubigeo = read_police_reports()

cargo_vehicles = read_cargo_vehicles()


ml_dataset = (enaho.merge(police_reports_by_ubigeo, on=['ubigeo', 'year'], how='left')
                    .merge(domestic_violence, on=['ubigeo', 'year', 'month'], how='left')
                    .merge(cargo_vehicles.drop(columns='quarter'), on=['ubigeo', 'year', 'month'], how='left')
                    )


ml_dataset.to_csv(os.path.join(dataPath, clean, 'ml_dataset_' + date +'.csv'))

