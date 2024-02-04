# Libraries
#--------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from joblib import load

#--------------
# Paths
#--------------


class DataPreparationForML:

    def __init__(self, freq='m', dataPath='J:/My Drive/PovertyPredictionRealTime/data', date='2023-11-14'):

        # 1. define data path:
        self.dataPath = dataPath

        # define paths:
        self.raw = '1_raw'

        self.working = '3_working'

        self.clean = '4_clean'

        # 2. define file names: 

        self.geo_sedlac = 'geo_sedlac_'
        self.onlygeo_sedlac = 'onlygeo_sedlac_'
        self.enaho = 'income.csv'
        self.enaho_sedlac = 'enaho_sedlac_pool_household.csv'
        self.domestic_violence = 'domestic_violence.csv'
        self.police_reports = 'delitos_distrito_comisaria_clean_panel.csv'
        self.domestic_violence = 'domestic_violence.csv'
        self.cargo_vehicles = 'cargo_vehicles.csv'
        self.planilla = 'labor.csv'

        self.temperature_max = '1_raw/peru/big_data/other/clima/Tmax.csv'
        self.temperature_min = '1_raw/peru/big_data/other/clima/Tmin.csv'
        self.nightlights_file = '1_raw/peru/big_data/other/nightlight/Nightlights_V01.csv'
        self.precipitation = '1_raw/peru/big_data/other/clima/Precipitation_V02.csv'
        self.panel_conglomerates = '1_raw/peru/data/conglomerado_centroidehogaresporconglomerado_2013-2021.csv' 

        # 3. define other parameters:

        # frequency
        self.freq = freq

        # define date of the file to read:
        self.date = date

        # 4. define dependent variable:
        self.depvar = 'log_income_pc'
        # self.depvar = 'log_income_pc_deviation'
        

        # 5. define independent variables:
        self.indepvar_enaho = ['log_income_pc_lagged', 'log_income_pc_lagged2', 'log_income_pc_lagged3', 'log_income_pc_lagged4']
        
        self.indepvar_enaho_missing = ['lag_missing', 
                                       'lag2_missing', 
                                       'lag3_missing', 
                                       'lag4_missing']
        # self.indepvar_enaho = ['log_income_pc_lagged'] #, 'log_income_pc_lagged2']

        self.indepvar_police_reports = ['Economic_Commercial_Offenses',
                                        'Family_Domestic_Issues', 'Fraud_Financial_Crimes',
                                        'Information_Cyber_Crimes', 'Intellectual_Property_Cultural_Heritage',
                                        'Miscellaneous_Offenses', 'Personal_Liberty_Violations',
                                        'Property_Real_Estate_Crimes', 'Public_Administration_Offenses',
                                        'Public_Order_Political_Crimes', 'Public_Safety_Health',
                                        'Sexual_Offenses', 'Theft_Robbery_Related_Crimes', 'Violence_Homicide']
        
        self.indepvar_planilla = ['companies',	'workers_tot',	'workers_sex_f_perc',	'workers_sex_m_perc', 
                                  'workers_sex_si_perc', 'workers_type_exe_perc', 'workers_type_wor_perc', 
                                  'workers_type_emp_perc',	'workers_type_nd_perc', 'salaries_mean']

        self.indepvar_precipitation =   ['Min_precipitation', 'Max_precipitation', 'Mean_precipitation',
                                            'Std_precipitation', 'Median_precipitation', 'Range_precipitation']

        self.indepvar_temperature_max =   ['Min_temperature_max', 'Max_temperature_max', 'Mean_temperature_max',
                                            'Std_temperature_max', 'Median_temperature_max', 'Range_temperature_max']        

        self.indepvar_temperature_min =   ['Min_temperature_min', 'Max_temperature_min', 'Mean_temperature_min', 
                                           'Std_temperature_min', 'Median_temperature_min', 'Range_temperature_min']

        self.indepvar_nightlights = ['min_nightlight','max_nightlight','mean_nightlight',
                                     'stdDev_nightlight','median_nightlight','range_nightlight']        

        self.indepvar_domestic_violence = ['cases_tot']

        self.indepvar_cargo_vehicles = ['vehicles_tot', 'fab_5y_p', 'fab_10y_p', 'fab_20y_p',
            'fab_30y_p', 'pub_serv_p', 'payload_m', 'dry_weight_m',
            'gross_weight_m', 'length_m', 'width_m', 'height_m']
        
        # self.indepvar_trend = ['trend' , 'trend2']
        self.indepvar_trend = ['trend']

        # self.indepvar_sample_selection = ['urbano','pondera_i']
        self.indepvar_sample_selection = []


        self.indepvar_lagged_income = self.indepvar_enaho + self.indepvar_enaho_missing

        self.indepvars = (self.indepvar_police_reports + #   self.indepvar_domestic_violence + 
                          self.indepvar_cargo_vehicles +
                          self.indepvar_planilla)
        
        self.indepvars_geodata = (self.indepvar_precipitation +
                                    self.indepvar_temperature_max +
                                    self.indepvar_temperature_min + 
                                    self.indepvar_nightlights)


    def obtain_ccpp_level_lags(self, enaho):
        """
        This function reads the enaho panel data and returns a dataframe with the following variables:

        - ubigeo: district code
        - year: year of the survey
        - month: month of the survey
        - conglome: conglomerate number
        - lagged income per capita up to 4 years.
        """
        enaho_conglome = enaho.copy()

        enaho_conglome['n_people'] = enaho_conglome['mieperho'] * enaho_conglome['pondera_i']

        household_weight = enaho_conglome['n_people']/enaho_conglome.groupby(['ubigeo','conglome', 'year'])['n_people'].transform('sum')

        # Get log income per capita weighted:
        enaho_conglome['log_income_pc'] = enaho_conglome['log_income_pc'] * household_weight

        enaho_conglome = (enaho_conglome.drop(columns=['dominio', 'estrato'])
                                        .groupby(['ubigeo','conglome', 'year'])
                                        .sum()
                                        .sort_values(by=['conglome', 'year'], ascending=True)
                                        .reset_index()
                                        )
        

        # First, ensure there's a row for every combination of 'conglome', and 'year' in the expected range
        all_years = range(enaho_conglome['year'].min(), enaho_conglome['year'].max() + 1)
        idx = pd.MultiIndex.from_product(
            [enaho_conglome['conglome'].unique(), all_years],
            names=['conglome', 'year']
        )

        enaho_conglome_full = enaho_conglome.set_index(['conglome', 'year']).reindex(idx).reset_index()

        # Get lagged income_pc ensuring that it corresponds to exactly 1-year lag
        for lag in range(1, 5):  # Adjust the range as needed
            enaho_conglome_full[f'log_income_pc_lagged{lag}'] = enaho_conglome_full.groupby(['ubigeo', 'conglome'])['log_income_pc'].shift(lag)

        # Drop rows that were artificially added and contain only NaNs except for 'ubigeo', 'conglome', and 'year'
        enaho_conglome = enaho_conglome_full.dropna(subset=['income_pc'])

        # Collapse the data to the conglome level:
        enaho_conglome = (enaho_conglome.sort_values(by=['conglome', 'year'])
                        .loc[:, ['ubigeo',
                                'conglome',
                                'year',
                                'log_income_pc_lagged1',
                                'log_income_pc_lagged2',
                                'log_income_pc_lagged3',
                                'log_income_pc_lagged4',
                                ]]
                        )
        
        return enaho_conglome
    

    def read_enaho_sedlac(self):

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


        # 1. Read csv
        enaho = pd.read_csv(os.path.join(self.dataPath, 
                                         self.working, 
                                         self.enaho_sedlac), index_col=0, parse_dates=True).reset_index()
        
        # 2. Manipulate identificator variables:
        enaho['ubigeo'] = 'U-' + enaho['ubigeo'].astype(str).str.zfill(6)
        enaho['year'] = enaho['year']

        # 3. Generate n_people to then compute average income per capita:
        enaho['n_people'] = enaho['mieperho'] * enaho['pondera_i']
        household_weight_year = enaho['n_people']/enaho.groupby(['year'])['n_people'].transform('sum')

        # 6. Compute income per capita (dependent variable):
        enaho['log_income_pc'] = np.log(enaho['income_pc']+0.1)

        # 4. Get demeaned version of log income per capita (this will be dependent variable):
        # basically what it does is log(income_pc) - \mu 
        enaho['log_income_pc_weighted'] = enaho['log_income_pc'] * household_weight_year
        enaho['log_income_pc_yearly_average'] = enaho.groupby(['year'])['log_income_pc_weighted'].transform('sum')
        enaho['log_income_pc_deviation'] = enaho['log_income_pc'] - enaho['log_income_pc_yearly_average']

        # 5. Get sum of income and individuals at the conglome:
        enaho_conglome = (self.obtain_ccpp_level_lags(enaho)
                                .rename(columns={'log_income_pc_lagged1':'log_income_pc_lagged'})
                            )

        enaho_conglome['lag_missing'] = enaho_conglome['log_income_pc_lagged'].isna().astype(int)
        enaho_conglome['lag2_missing'] = enaho_conglome['log_income_pc_lagged2'].isna().astype(int)
        enaho_conglome['lag3_missing'] = enaho_conglome['log_income_pc_lagged3'].isna().astype(int)
        enaho_conglome['lag4_missing'] = enaho_conglome['log_income_pc_lagged4'].isna().astype(int)

        # 7. Get conglome data to enaho:
        enaho = (enaho.merge(enaho_conglome, on=['ubigeo','conglome', 'year'], how='left')
                        .rename(columns={'mes':'month'})
                        )

        return enaho


    def read_enaho_sedlac_ccpp(self):

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
        enaho = pd.read_csv(os.path.join(self.dataPath, 
                                         self.working, 
                                         self.enaho_sedlac), index_col=0, parse_dates=True).reset_index()

        # Manipulate identificator variables:
        enaho['ubigeo'] = 'U-' + enaho['ubigeo'].astype(str).str.zfill(6)
        enaho['year'] = enaho['year']

        # Get sum of income and individuals at the conglome:

        enaho_conglome = self.obtain_ccpp_level_lags(enaho)

        return enaho_conglome
    


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

        return cargo_vehicles


    def read_planilla_electronica(self):
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

        planilla = pd.read_csv(os.path.join(self.dataPath, self.working, self.planilla), index_col=0).reset_index()

        planilla['ubigeo'] = 'U-' + planilla['ubigeo'].astype(str).str.zfill(6)

        return planilla


    def read_precipitation(self):

        """
        This function reads the precipitation data and returns a dataframe with the following variables:s
        """

        #--------------
        # Opening main data
        #--------------
        df = pd.read_csv(os.path.join(self.dataPath, self.precipitation), encoding='iso-8859-1', on_bad_lines='skip')

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

        precipitation = df.copy() 
        precipitation = precipitation.drop('Month', axis=1)

        #--------------
        # Renaming
        #--------------
        precipitation.rename(columns={'Conglomerado ID': 'conglome'}, inplace=True)

        precipitation.columns = [col  
                                 if col in ['conglome','year','month'] 
                                 else col + '_precipitation' 
                                 for col in precipitation.columns
                                 ]


        return precipitation
    

    def read_nightlights(self):

        """
        This function reads the precipitation data and returns a dataframe with the following variables:s
        """

        #--------------
        # Opening main data
        #--------------
        df = pd.read_csv(os.path.join(self.dataPath, self.nightlights_file), encoding='iso-8859-1', on_bad_lines='skip')

        #--------------
        # Creating variables
        #--------------
        # year
        df['year'] = ((df['month'] - 1) // 12) + 2014

        # month
        df['month'] = df['month'] % 12
        df.loc[df['month'] == 0, 'month'] = 12

        #--------------
        # Droping
        #--------------

        nightlights = df.copy() 
        # nightlights = nightlights.drop('Month', axis=1)

        #--------------
        # Renaming
        #--------------
        nightlights.rename(columns={'Conglomerado ID': 'conglome'}, inplace=True)

        nightlights.columns = [col  
                            if col in ['conglome','year','month'] 
                            else col + '_nightlight' 
                            for col in nightlights.columns
                            ]


        return nightlights


    def read_min_temperature(self):

        """
        This function reads the precipitation data and returns a dataframe with the following variables:s
        """

        #--------------
        # Opening main data
        #--------------
        df = pd.read_csv(os.path.join(self.dataPath, self.temperature_min), encoding='iso-8859-1', on_bad_lines='skip')

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

        t_min = df.copy() 
        t_min = t_min.drop('Month', axis=1)

        #--------------
        # Renaming
        #--------------
        t_min.rename(columns={'Conglomerado ID': 'conglome'}, inplace=True)

        t_min.columns = [col  
                            if col in ['conglome','year','month'] 
                            else col + '_temperature_min' 
                            for col in t_min.columns
                            ]


        return t_min
    
    
    def read_max_temperature(self):

        """
        This function reads the precipitation data and returns a dataframe with the following variables:s
        """

        #--------------
        # Opening main data
        #--------------
        df = pd.read_csv(os.path.join(self.dataPath, self.temperature_max), encoding='iso-8859-1', on_bad_lines='skip')

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

        t_max = df.copy() 
        t_max = t_max.drop('Month', axis=1)

        #--------------
        # Renaming
        #--------------
        t_max.rename(columns={'Conglomerado ID': 'conglome'}, inplace=True)

        t_max.columns = [col  
                            if col in ['conglome','year','month'] 
                            else col + '_temperature_max' 
                            for col in t_max.columns
                            ]


        return t_max    

    def read_conglome_clusters(self):

        """
        This function reads the conglome clusters and returns a dataframe with the following variables:
        """

        df = pd.read_csv(os.path.join(self.dataPath, self.panel_conglomerates), encoding='iso-8859-1', on_bad_lines='skip')

        #--------------
        # Cleaning column names
        #--------------

        # Dictionary of matching column names & new column names
        name_mapping = {
            'conglome' : 'conglome',
            'ubigeo'   : 'ubigeo',
            'mes'      : 'month',
            'ano'      : 'year'
        }

        # Map function to rename old matching names with new column names
        # Applying the map function to rename variables based on the dictionary
        df = df.rename(columns=name_mapping)

        #--------------
        # Filtering columns
        #--------------
        df = df[list(name_mapping.values())]

        #--------------
        # Dropping duplicates
        #--------------
        conglomerates = df.drop_duplicates().copy()


        return conglomerates



    def read_consolidated_ml_dataset(self):

        """
        This function reads the consolidated ml dataset and returns a dataframe:
        """
        
        ml_dataset = pd.read_csv(os.path.join(self.dataPath, self.clean, 'ml_dataset_' + self.date +'.csv'), index_col=0)


        return ml_dataset



    def filter_ml_dataset(self, ml_dataset):
        """
        Filter the dataset to only include the observations that have all the years
        Parameters
        ----------
        ml_dataset : dataframe
        """
        ml_dataset = ml_dataset.query('income_pc>0')
        
        # First pass dropping all missing values:
        ml_dataset_filtered = (ml_dataset.query('year >= 2014')
                                        .query('year <= 2019')
                                        .sample(frac=1) # Random shuffle
                                        .reset_index(drop=True) # Remove index
                                        )
        ml_dataset_filtered['count_people'] = 1
        conglome_count = ml_dataset_filtered.groupby(['conglome','year']).count().reset_index().loc[:,['conglome','year','count_people']]
        conglome_count['count'] = conglome_count.groupby(['conglome']).transform('count')['year']

        # Filter out conglomerates that do not have all the years:        
        ml_dataset_filtered = ml_dataset_filtered.dropna(subset='log_income_pc_lagged').reset_index(drop=True)
        return ml_dataset_filtered


    def input_missing_values(self, ml_dataset_filtered):

        # Define the independent variables to be used in the model:
        indepvar_column_names = self.indepvars + self.indepvars_geodata + self.indepvar_sample_selection

        # Define dependent and independent variables:
        X = ml_dataset_filtered.loc[:,self.indepvar_lagged_income + indepvar_column_names + self.indepvar_trend].copy()
        X[indepvar_column_names] = np.log(X[indepvar_column_names] + 1)
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        ml_dataset_filtered.loc[:,self.indepvar_lagged_income + indepvar_column_names + self.indepvar_trend] = X_imputed

        return ml_dataset_filtered


    def get_depvar_and_features(self, ml_dataset_filtered, scaler_X=None, scaler_Y=None, interaction=True):
        """
        Get the training sample
        Parameters
        ----------
        ml_dataset_filtered : dataframe
        """

        # Define the independent variables to be used in the model:
        indepvar_column_names = self.indepvars + self.indepvars_geodata + self.indepvar_sample_selection

        # Define dependent and independent variables:
        Y = ml_dataset_filtered.loc[:,self.depvar].reset_index(drop=True) 
        X = ml_dataset_filtered.loc[:,self.indepvar_lagged_income + indepvar_column_names + self.indepvar_trend]

        if scaler_X is None:  #(This is to demean test data)
            # Step 2: Standardize X
            scaler_X = StandardScaler()
            X_standardized = scaler_X.fit_transform(X)
            X_standardized = pd.DataFrame(X_standardized, columns=X.columns)
            # Step 3: Standardize Y
            scaler_Y = StandardScaler()
            Y_standardized = pd.Series(scaler_Y.fit_transform(Y.values.reshape(-1, 1)).flatten())  # Use flatten to convert it back to 1D array
        else:
            X_standardized = scaler_X.transform(X)
            X_standardized = pd.DataFrame(X_standardized, columns=X.columns)
            Y_standardized = pd.Series(scaler_Y.transform(Y.values.reshape(-1, 1)).flatten())
        
        # Step 4: Generate dummy variables for ubigeo and month: 
        ubigeo_dummies = pd.get_dummies(ml_dataset_filtered['ubigeo'].str[:4], prefix='ubigeo', drop_first=True).reset_index(drop=True)
        month_dummies = pd.get_dummies(ml_dataset_filtered['month'], prefix='month', drop_first=True).reset_index(drop=True)
        area_dummies = pd.get_dummies(ml_dataset_filtered['strata'], prefix='strata', drop_first=True).reset_index(drop=True)
        
        # Step 5: Adding the dummy variables to X
        X_standardized = pd.concat([X_standardized, 
                                    ubigeo_dummies.astype(int), 
                                    month_dummies.astype(int), 
                                    area_dummies.astype(int)], axis=1)

        if interaction == True:
            # Step 6: Create interaction terms:
            variables_to_interact = self.indepvar_lagged_income + indepvar_column_names
            # Create interaction terms
            for var in variables_to_interact:
                
                for dummy in ubigeo_dummies.columns:
                    interaction_term = X_standardized[var] * ubigeo_dummies[dummy]
                    X_standardized[f"{var}_x_{dummy}"] = interaction_term

                for dummy in area_dummies.columns:
                    interaction_term = X_standardized[var] * area_dummies[dummy]
                    X_standardized[f"{var}_x_{dummy}"] = interaction_term

        # Step 7: Split the model in validation data and train and testing data:        
        Y_standardized_train = Y_standardized
        X_standardized_train = X_standardized
        X_standardized_train['const'] = 1

        return Y_standardized_train, X_standardized_train, scaler_X, scaler_Y


    def load_ml_model(self, model_filename = 'best_weighted_lasso_model.joblib'):
        """
        This function loads the best model from the specified file.
        """

        best_model_loaded = load(model_filename)
        
        return best_model_loaded


#%% Run the code:

if __name__ == '__main__':

    dataPath = 'J:/My Drive/PovertyPredictionRealTime/data'

    freq = 'm'

    date = datetime.today().strftime('%Y-%m-%d')

    #--------------

    dpml = DataPreparationForML(freq=freq, dataPath=dataPath, date=date)

    # Load data:

    # enaho_ccpp = dpml.read_enaho_sedlac_ccpp()

    enaho = dpml.read_enaho_sedlac()

    domestic_violence = dpml.read_domestic_violence_cases()

    police_reports_by_ubigeo = dpml.read_police_reports()

    cargo_vehicles = dpml.read_cargo_vehicles()
    
    labor = dpml.read_planilla_electronica()

    precipitation = dpml.read_precipitation()

    nightlights = dpml.read_nightlights()
    
    max_temperature = dpml.read_max_temperature()
    
    min_temperature = dpml.read_min_temperature()

    # Merge data:
    ml_dataset = (enaho.merge(police_reports_by_ubigeo, on=['ubigeo', 'year'], how='left')
                        .merge(domestic_violence, on=['ubigeo', 'year', 'month'], how='left')
                        .merge(labor, on=['ubigeo', 'year', 'month'], how='left')
                        .merge(cargo_vehicles.drop(columns='quarter'), on=['ubigeo', 'year', 'month'], how='left')
                        .merge(precipitation, on=['conglome', 'year', 'month'], how='left')
                        .merge(nightlights, on=['conglome', 'year', 'month'], how='left')
                        .merge(max_temperature, on=['conglome', 'year', 'month'], how='left')
                        .merge(min_temperature, on=['conglome', 'year', 'month'], how='left')
                        )
    
    # Add trend and trend squared:
    ml_dataset['trend'] = ml_dataset['year'].astype(int) - 2011

    
    ml_dataset['trend2'] = ml_dataset['trend']**2

    ml_dataset.to_csv(os.path.join(dpml.dataPath, dpml.clean, 'ml_dataset_' + date +'.csv'))


    # Get conglome pool (These are centroids to get weather data):
    conglome_panel = (enaho[['conglome', 'ubigeo', 'strata', 'latitud', 'longitud', 'year']]
                      .groupby(['conglome', 'ubigeo', 'strata','year'])
                      .mean()
                      .reset_index()
                      )
    
    conglome_panel['centroid_id'] = conglome_panel['ubigeo'].astype(str) + '_' + conglome_panel['conglome'].astype(str).str.zfill(5) + '_' + conglome_panel['year'].astype(str)

    repeated_conglome_panel = pd.concat([conglome_panel.assign(month=i) for i in range(1, 13)], ignore_index=True)

    repeated_conglome_panel.to_csv(os.path.join(dpml.dataPath, dpml.raw,'peru/data', 'conglome_panel_final_version.csv'), index=False)



    #%% Some checks on the data for lag incme at CCPP:

    # enaho_ccpp['missing_lag1'] = enaho_ccpp['log_income_pc_lagged1'].isna()
    # enaho_ccpp['missing_lag2'] = enaho_ccpp['log_income_pc_lagged2'].isna()
    # enaho_ccpp['missing_lag3'] = enaho_ccpp['log_income_pc_lagged3'].isna()
    # enaho_ccpp['missing_lag4'] = enaho_ccpp['log_income_pc_lagged4'].isna()

    # enaho_ccpp.groupby('year')[['missing_lag1',
    #                             'missing_lag2',
    #                             'missing_lag3',
    #                             'missing_lag4']].mean().round(3).to_csv(os.path.join('..','tables', 'missing_lags.csv'))
    # enaho_ccpp[['missing_lag1',
    #             'missing_lag2',
    #             'missing_lag3',
    #             'missing_lag4']].mean()