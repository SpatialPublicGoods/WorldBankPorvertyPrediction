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

    for yy in range(2008, 2021):

        print('Reading ENAHO data for year: ', yy)
        
        if yy >= 2013:
            enaho_yy = pd.read_stata(os.path.join(dpml.dataPath, 
                                                dpml.raw,
                                                'peru/data/SEDLAC - ENAHO Merged Data/',
                                                dpml.geo_sedlac + str(yy) + '.dta')).rename(columns={'ano_ocaux':'year'})
        else:
            enaho_yy = pd.read_stata(os.path.join(dpml.dataPath, 
                                                dpml.raw,
                                                'peru/data/SEDLAC - ENAHO Merged Data/',
                                                dpml.onlygeo_sedlac + str(yy) + '.dta')).rename(columns={'ano_ocaux':'year'})
        
        # Filter out variables and obtain household level data:
        enaho_yy = filter_variables_and_obtain_household_level_data(enaho_yy)

        # Append to list:
        list_enaho_datasets.append(enaho_yy)

    # Concatenate all datasets:       
    enaho_sedlac_panel = pd.concat(list_enaho_datasets, axis=0)

    enaho_sedlac_panel.to_csv(os.path.join(dpml.dataPath,
                                           dpml.working, 
                                           'enaho_sedlac_pool_household.csv'), index=False)

    return enaho_sedlac_panel



def filter_variables_and_obtain_household_level_data(enaho_sedlac):
    """
    Filter out variables and obtain household level data

    Parameters
    ----------
    enaho_sedlac : Dataframe with individual level data.
    """

    # Keep only relevant columns:
    enaho_sedlac_filtered = enaho_sedlac.loc[:,['ubigeo','conglome','dominio', 'estrato',
                                                'vivienda','hogar_ine','strata', 
                                                'year', 'mes', 'latitud','longitud',
                                                'mieperho', 'ipcf_ppp17','lp_215usd_ppp',
                                                'lp_365usd_ppp','lp_685usd_ppp','pondera_i']]

    # Filter out missing values:
    enaho_sedlac_filtered_household = (enaho_sedlac_filtered.groupby(['ubigeo','conglome','vivienda','hogar_ine','year', 'mes'])
                                                        .first()
                                                        .reset_index()
                                                        .rename(columns={'ipcf_ppp17':'income_pc',
                                                                        'mes':'month'})
                                                        )
    
    return enaho_sedlac_filtered_household
    


# Run function:

enaho_sedlac_panel = append_enaho_sedlac()




