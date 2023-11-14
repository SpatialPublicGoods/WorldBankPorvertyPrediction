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


class DataPreparationForML:

    def __init__(self, freq='m', dataPath='J:/My Drive/PovertyPredictionRealTime/data', date='2023-11-14'):

        # 1. define data path:
        self.dataPath = dataPath

        # define paths:
        self.working = '3_working'

        self.clean = '4_clean'

        # 2. define file names: 

        self.enaho = 'income.csv'
        self.domestic_violence = 'domestic_violence.csv'
        self.police_reports = 'delitos_distrito_comisaria_clean_panel.csv'
        self.domestic_violence = 'domestic_violence.csv'
        self.cargo_vehicles = 'cargo_vehicles.csv'


        # 3. define other parameters:

        # frequency
        self.freq = freq

        # define date of the file to read:
        self.date = date

        # 4. define dependent variable:
        self.depvar = 'income_pc'
        

        # 5. define independent variables:
        self.indepvar_enaho = ['income_pc_lagged','spend_pc_lagged']

        self.indepvar_police_reports = ['Economic_Commercial_Offenses',
                                        'Family_Domestic_Issues', 'Fraud_Financial_Crimes',
                                        'Information_Cyber_Crimes', 'Intellectual_Property_Cultural_Heritage',
                                        'Miscellaneous_Offenses', 'Personal_Liberty_Violations',
                                        'Property_Real_Estate_Crimes', 'Public_Administration_Offenses',
                                        'Public_Order_Political_Crimes', 'Public_Safety_Health',
                                        'Sexual_Offenses', 'Theft_Robbery_Related_Crimes', 'Violence_Homicide']

        self.indepvar_domestic_violence = ['cases_tot']

        self.indepvar_cargo_vehicles = ['vehicles_tot', 'fab_5y_p', 'fab_10y_p', 'fab_20y_p',
            'fab_30y_p', 'pub_serv_p', 'payload_m', 'dry_weight_m',
            'gross_weight_m', 'length_m', 'width_m', 'height_m']

        self.indepvars = self.indepvar_enaho + self.indepvar_police_reports + self.indepvar_domestic_violence + self.indepvar_cargo_vehicles


    def read_enaho_panel(self):

        """
        This function reads the enaho panel data and returns a dataframe with the following variables:
        
        - ubigeo: district code
        - year: year of the survey
        - month: month of the survey
        - conglome: conglomerate number
        - mieperho: number of individuals in the household
        - ingmo1hd: household income
        - gashog1d: household expenditure
        - income_pc: household income per capita
        - log_income_pc: log of household income per capita
        - spend_pc: household expenditure per capita
        - log_spend_pc: log of household expenditure per capita
        - income_pc_lagged: lagged income per capita
        - spend_pc_lagged: lagged expenditure per capita

        """

        # Read csv
        enaho = pd.read_csv(os.path.join(self.dataPath, self.working, self.enaho), index_col=0, parse_dates=True).reset_index()

        enaho['ubigeo'] = 'U-' + enaho['ubigeo'].astype(str).str.zfill(6)

        enaho['year'] = enaho['year'].dt.year

        # Get sum of income and individuals at the conglome:
        enaho_conglome = enaho.groupby(['conglome', 'year']).sum().reset_index()

        # Compute income per capita:

        enaho_conglome['income_pc'] = enaho_conglome['ingmo1hd']/enaho_conglome['mieperho']

        enaho_conglome['log_income_pc'] = np.log(enaho_conglome['income_pc']+0.1)

        # Compute gasto per capita:

        enaho_conglome['spend_pc'] = enaho_conglome['gashog1d']/enaho_conglome['mieperho']

        enaho_conglome['log_spend_pc'] = np.log(enaho_conglome['income_pc']+0.1)

        # Get lagged income_pc
        enaho_conglome['income_pc_lagged'] = enaho_conglome.groupby('conglome')['income_pc'].shift(1)
        enaho_conglome['spend_pc_lagged'] = enaho_conglome.groupby('conglome')['spend_pc'].shift(1)

        enaho_conglome = (enaho_conglome.sort_values(by=['conglome', 'year'])
                        .loc[:, ['conglome','year','log_income_pc','log_spend_pc','income_pc','income_pc_lagged','spend_pc','spend_pc_lagged']]
                        )
        
        # Get conglome data to enaho:
        enaho = enaho.merge(enaho_conglome, on=['conglome', 'year'], how='left')

        return enaho


    def read_domestic_violence_cases(self):
        
        """
        This function reads the domestic violence cases and returns a dataframe with the following variables:

        - ubigeo: district code
        - year: year of the survey
        - month: month of the survey
        - cases_tot: total number of cases

        """

        domestic_violence = pd.read_csv(os.path.join(self.dataPath, self.working, self.domestic_violence), index_col=0).reset_index()

        domestic_violence['ubigeo'] = 'U-' + domestic_violence['ubigeo'].astype(str).str.zfill(6)

        domestic_violence = domestic_violence.groupby(['ubigeo','year', 'month']).first().reset_index()

        return domestic_violence


    def read_police_reports(self):

        """
        This function reads the police reports and returns a dataframe with the following variables:

        - ubigeo: district code
        - year: year of the survey
        - month: month of the survey
        - Economic_Commercial_Offenses: number of cases
        - Family_Domestic_Issues: number of cases
        - Fraud_Financial_Crimes: number of cases
        - Information_Cyber_Crimes: number of cases
        - Intellectual_Property_Cultural_Heritage: number of cases
        - Miscellaneous_Offenses: number of cases
        - Personal_Liberty_Violations: number of cases
        - Property_Real_Estate_Crimes: number of cases
        ....
        - Violence_Homicide: number of cases
        """

        police_reports = pd.read_csv(os.path.join(self.dataPath, 
                                                  self.working, 
                                                  self.police_reports), 
                                                  index_col=0).reset_index()

        police_reports['ubigeo'] = 'U-' + police_reports['ubigeo'].astype(str).str.zfill(6)

        police_reports['year'] = police_reports['Year']

        # Group by granularity and frequency:
        police_reports_by_ubigeo = police_reports.groupby(['year', 'ubigeo']).sum().reset_index()

        return police_reports_by_ubigeo



    def read_cargo_vehicles(self):
        """
        This function reads the cargo vehicles and returns a dataframe with the following variables:

        - ubigeo: district code
        - year: year of the survey
        - month: month of the survey
        - vehicles_tot: total number of vehicles
        - fab_5y_p: percentage of vehicles fabricated in the last 5 years
        - fab_10y_p: percentage of vehicles fabricated in the last 10 years
        - fab_20y_p: percentage of vehicles fabricated in the last 20 years
        - fab_30y_p: percentage of vehicles fabricated in the last 30 years
        - pub_serv_p: percentage of vehicles for public service
        - payload_m: payload in metric tons
        - dry_weight_m: dry weight in metric tons
        - gross_weight_m: gross weight in metric tons
        - length_m: length in meters
        - width_m: width in meters
        - height_m: height in meters
        - quarter: quarter of the year

        """

        cargo_vehicles = pd.read_csv(os.path.join(self.dataPath, self.working, self.cargo_vehicles), index_col=0).reset_index()

        cargo_vehicles['ubigeo'] = 'U-' + cargo_vehicles['ubigeo'].astype(str).str.zfill(6)


    def read_consolidated_ml_dataset(self):

        """
        This function reads the consolidated ml dataset and returns a dataframe:
        """
        
        ml_dataset = pd.read_csv(os.path.join(self.dataPath, self.clean, 'ml_dataset_' + self.date +'.csv'), index_col=0)


        return ml_dataset


#%% Run the code:

if __name__ == '__main__':

    dataPath = 'J:/My Drive/PovertyPredictionRealTime/data'

    freq = 'm'

    date = datetime.today().strftime('%Y-%m-%d')

    #--------------

    dpml = DataPreparationForML(freq=freq, dataPath=dataPath, date=date)

    # Load data:

    enaho = dpml.read_enaho_panel()

    domestic_violence = dpml.read_domestic_violence_cases()

    police_reports_by_ubigeo = dpml.read_police_reports()

    cargo_vehicles = dpml.read_cargo_vehicles()


    ml_dataset = (enaho.merge(police_reports_by_ubigeo, on=['ubigeo', 'year'], how='left')
                        .merge(domestic_violence, on=['ubigeo', 'year', 'month'], how='left')
                        .merge(cargo_vehicles.drop(columns='quarter'), on=['ubigeo', 'year', 'month'], how='left')
                        )


    ml_dataset.to_csv(os.path.join(dpml.dataPath, dpml.clean, 'ml_dataset_' + date +'.csv'))

