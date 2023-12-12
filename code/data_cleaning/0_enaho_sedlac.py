import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from consolidate_ml_dataframe import DataPreparationForML

#--------------

dataPath = 'J:/My Drive/PovertyPredictionRealTime/data'

freq = 'm'

date = datetime.today().strftime('%Y-%m-%d')

#--------------


dpml = DataPreparationForML(freq=freq, dataPath=dataPath, date=date)


def append_enaho_sedlac():

    """
    Append sedlac data
    """
    
    list_enaho_datasets = []

    for yy in range(2013, 2021):
                
        enaho_yy = pd.read_stata(os.path.join(dpml.dataPath, 
                                            dpml.raw,
                                            'peru/data/SEDLAC - ENAHO Merged Data/',
                                            dpml.enaho_sedlac + str(yy) + '.dta'))

        
        list_enaho_datasets.append(enaho_yy)
        
    enaho_sedlac_panel = pd.concat(list_enaho_datasets, axis=0)

    enaho_sedlac_panel.to_csv(os.path.join(dpml.dataPath,dpml.working, 
                                           'enaho_sedlac_panel.csv'), index=False)

    return enaho_sedlac_panel


def read_appended_enaho_sedlac():

    enaho_sedlac_pool = pd.read_csv(os.path.join(dpml.dataPath,dpml.working, 'enaho_sedlac_panel.csv'), index_col=False)

    enaho_sedlac_pool.rename(columns={'ano_ocaux':'year'}, inplace=True)

    return enaho_sedlac_pool


# Read enaho sedlac
enaho_sedlac_pool = read_appended_enaho_sedlac()

# Keep only relevant columns:
enaho_sedlac_pool_filtered = enaho_sedlac_pool.loc[:,['ubigeo','conglome','vivienda','hogar_ine','strata', 
                                             'year', 'mes', 'latitud','longitud',
                                            'mieperho', 'ipcf_ppp17','lp_215usd_ppp',
                                            'lp_365usd_ppp','lp_685usd_ppp','pondera_i']]

# Filter out missing values:
enaho_sedlac_pool_household = (enaho_sedlac_pool_filtered.groupby(['ubigeo','conglome','vivienda','hogar_ine','year', 'mes'])
                                                            .first()
                                                            .reset_index()
                                                            .rename(columns={'ipcf_ppp17':'income_pc',
                                                                             'mes':'month'})
                                                            )

# Save to csv:
enaho_sedlac_pool_household.to_csv(os.path.join(dpml.dataPath,
                                                dpml.working, 
                                                'enaho_sedlac_pool_household.csv'), index=False)
