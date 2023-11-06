import os
from re import S
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import jaro
import json
from .data_column_names import columname # 
from .utils_general import utils_general

class dataManipulationENDES:

    def __init__(self, dataPath):

        # 1. Global string variables for paths and relative paths:
        self.dataPath =  dataPath
        self.raw = "raw"
        self.intermediate = "intermediate"
        self.working = "working"
        self.clean = "clean"

        self.endes = "ENDES"

        # Load column name class which is stored in another file...
        
        self.columname = columname() 
        self.utils_general = utils_general(self.dataPath)
        self.recoding_dict = self.generate_recoding_dictionary()


    def generate_recoding_dictionary(self):

        recoding_dict = {}

        recoding_dict['B5'] = { 'Si': 'Yes',
                                'Sí': 'Yes',
                                'Yes':'Yes',
                                'No':'No'
                                }

        recoding_dict['SH52'] = {"Public agency/company":"public agency",
                                "JASS":"JASS",
                                "Agencia o compañía pública":"public agency",
                                "Junta Administradora de los Servicios de Saneamiento (JASS)":"JASS",
                                "Other" :"Other",
                                "Otro" :"Other",
                                "Private agency/company" : "private agency",
                                "Agencia o compañía privada":"private agency",
                                "Other private water provider":"other private provider",
                                "Otro proveedor de agua privado":"other private provider"
                                }

        recoding_dict['SH227'] = {'0,0 mg/lt':'0.0 mg/lt',
                                '0.0 mg/Lt.':'0.0 mg/lt',
                                '0.0 mg/lt':'0.0 mg/lt',
                                '>=  0.5 mg/lt':'>=  0.5 mg/lt',
                                '>= 0,5 mg/lt':'>=  0.5 mg/lt',
                                'Mayor o Igual a 0.5 mg/Lt.':'>=  0.5 mg/lt',
                                'De 0,1 a < de 0,5 mg/lt':'De 0.1 a menos de 0.5 mg/lt',
                                'De 0,1 a menos de 0,5 mg/lt':'De 0.1 a menos de 0.5 mg/lt',
                                'De 0.1 mg/Lt. A menos de 0.5 mg/Lt.':'De 0.1 a menos de 0.5 mg/lt',
                                'From 0.1 to < than 0.5 mg/lt':'De 0.1 a menos de 0.5 mg/lt',
                                'No se pudo realizar la Prueba':'not tested',
                                'No se pudo tomar la prueba':'not tested',
                                'No se tomó el test':'not tested',
                                'Test could not be taken':'not tested',
                                'Take packed water':'packed water',
                                'Toma agua embotellada':'packed water',
                                'Toman agua embotellada':'packed water',
                                'Take the water as it comes from river, spring, well, etc':'natural source',
                                'Toma el agua como viene del rio, manantial, pozo, etc':'natural source',
                                'Toma el agua tal como viene del rio, manantial, pozo, etc':'natural source',
                                'La toman tal como viene del: Rio, Acequia, Pozo, etc.':'natural source'
                                }

        recoding_dict['HV201'] = {
                                'Piped into dwelling':'Piped into dwelling', 
                                'Red dentro de vivienda':'Piped into dwelling', 
                                'Dentro de la vivienda':'Piped into dwelling', 

                                'Piped outside dwelling but within buikding':'Piped into building',
                                'Red fuera de la vivienda pero dentro de la edificación':'Piped into building',
                                'Fuera de la vivienda, pero dentro del edificio':'Piped into building',

                                'Public tap/standpipe':'Public tap', 
                                'Pilón/Grifo público':'Public tap',                                                  
                                'Pilón, grifo público':'Public tap', 

                                'Pozo dentro de vivienda':'Well inside dwelling',                                           
                                "Well inside dwelling":'Well inside dwelling', 
                                'Pozo en la vivineda/patio/lote':'Well inside dwelling',

                                'Pozo público':'Public well',
                                'Public well':'Public well',

                                'Bottled water':'Bottled water', 
                                'Agua embotellada':'Bottled water', 

                                'River/dam/lake/ponds/stream/canal/irirgation channel':'River', 
                                'Rió, presa, lago,estanque, arroyo, canal o canal de irrigación':'River', 
                                'Río/acequia/laguna':'River',                                                  
                                'Río, presa, lago,estanque, arroyo, canal o canal de irrigación':'River',      

                                'Manantial (puquio)':'Spring',                                                   
                                'Spring':'Spring',
                                'Manantial':'Spring',   

                                'Tanker truck':'truck',  
                                'Camión cisterna':'truck',             

                                'Other':'Other', 
                                'Otro':'Other',  

                                'Rainwater':'Rain',                                                            
                                'Agua de lluvia':'Rain'                                                      
        }


        return recoding_dict


    # Functions to read dataframes

    def read_household_data(self, year):

        """
        This function reads the household data from the ENDES survey.
        
        Parameters
        ----------
        year : int
            Year of the survey.
            
        Returns
        -------
        household_data : pandas dataframe
            Dataframe with the household data.
            
        """

        # 1. Generate file name
        filename = os.path.join(self.dataPath, self.raw, self.endes, str(year), "RECH0.SAV")

        # 2. Read the data
        household_data = pd.read_spss(filename)

        return household_data

    

    def read_health_data(self, year):
        """
        
        This function reads the household health data from the ENDES survey.
        
        Parameters
        ----------
        year : int
            Year of the survey.
                    
        Returns
        -------
        household_health_data : pandas dataframe
            Dataframe with the household health data.
        
        """

        # 1. Generate file name
        filename = os.path.join(self.dataPath, self.raw, self.endes, str(year), "RECH23.SAV")

        # 2. Read the data
        household_health_data = pd.read_spss(filename)

        return household_health_data


    def read_maternity_data(self, year):
        """
        
        This function reads the household health data from the ENDES survey.
        
        Parameters
        ----------
        year : int
            Year of the survey.
                    
        Returns
        -------
        household_health_data : pandas dataframe
            Dataframe with the household health data.
        
        """

        # 1. Generate file name
        filename = os.path.join(self.dataPath, self.raw, self.endes, str(year), "REC21.SAV")

        # 2. Read the data
        maternity_data = pd.read_spss(filename)

        # 3. Recode "Child is alive" question
        maternity_data['B5'] = [self.recoding_dict['B5'][x] if pd.notna(x) else x  for x in maternity_data['B5']] 

        return maternity_data


    def obtain_household_panel_data(self, year_list):

        """
        This function reads the household data from the ENDES survey.

        Parameters
        ----------
        year_list : list
            List of years of the survey.

        Returns
        -------
        household_panel_data : pandas dataframe
            Dataframe with the household data.

        """

        household_data_list = []

        for year in year_list:

            household_data = self.read_household_data(year)

            household_data['year'] = year 

            household_data_list.append(household_data)

        household_panel_data = pd.concat(household_data_list)

        # Consolidate variables:
        household_panel_data = self.consolidate_variables(household_panel_data, ['longitudx', 'long_ccpp'], 'longitud_ccpp')

        household_panel_data = self.consolidate_variables(household_panel_data, ['latitudy', 'lat_ccpp'], 'latitud_ccpp')

        household_panel_data = self.consolidate_variables(household_panel_data, ['codccpp', 'CODCCPP'], 'cod_ccpp')

        household_panel_data = self.consolidate_variables(household_panel_data, ['nomccpp', 'NOMCCPP'], 'nom_ccpp')        

        household_panel_data['HHID'] = household_panel_data['HHID'].str.replace(' ','')


        return household_panel_data


    def obtain_health_panel_data(self, year_list):

        """
        This function reads the household health data from the ENDES survey.
        
        Parameters
        ----------
        year_list : list
            List of years of the survey.
            
        Returns
        -------
        health_panel_data : pandas dataframe
            Dataframe with the household health data.
        
        """

        health_data_list = []

        for year in year_list:

            health_data = self.read_health_data(year)

            health_data['year'] = year

            if year == 2010:

                health_data = health_data.drop(columns=['SH51', 'SH52'])

                health_data = health_data.rename(columns={'SH49':'SH51', # pay or not (stop asking 2016)
                                                            'SH50':'SH52', # agency
                                                            'SH127':'SH227' # chlorination
                                                            })

            health_data_list.append(health_data)

        health_panel_data = pd.concat(health_data_list)

        # Remove empty spaces in hhid
        health_panel_data['HHID'] = health_panel_data['HHID'].str.replace(' ','')

        # Recode chlorine data:
        health_panel_data['SH227'] = [self.recoding_dict['SH227'][x] if pd.notna(x) else x  for x in health_panel_data['SH227']] # health_panel_data['227'] = [self.recoding_dict['227'][x] for x in health_panel_data['227']]

        health_panel_data['HV201'] = [self.recoding_dict['HV201'][x] if pd.notna(x) else x  for x in health_panel_data['HV201']] # health_panel_data['227'] = [self.recoding_dict['227'][x] for x in health_panel_data['227']]

        health_panel_data['HV202'] = [self.recoding_dict['HV201'][x] if pd.notna(x) else x  for x in health_panel_data['HV202']] # health_panel_data['227'] = [self.recoding_dict['227'][x] for x in health_panel_data['227']]

        return health_panel_data




    def obtain_maternity_panel_data(self, year_list):

        """
        This function reads the household health data from the ENDES survey.
        
        Parameters
        ----------
        year_list : list
            List of years of the survey.
            
        Returns
        -------
        health_panel_data : pandas dataframe
            Dataframe with the household health data.
        
        """

        maternity_data_list = []

        for year in year_list:

            maternity_data = self.read_maternity_data(year)

            maternity_data['year'] = year

            maternity_data_list.append(maternity_data)

        maternity_panel_data = pd.concat(maternity_data_list)

        # Obtain Household ID out of CASEID
        maternity_panel_data['HHID'] = [' '.join(x.split(' ')).split()[0] for x in maternity_panel_data['CASEID']]

        maternity_panel_data['HHID'] = maternity_panel_data['HHID'].str.replace(' ','')


        return maternity_panel_data


    def obtain_endes_consolidated_panel_data(self, year_list):

        """
        This function reads the household health data from the ENDES survey.
        
        Parameters
        ----------
        year_list : list
            List of years of the survey.
            
        Returns
        -------
        merged_dataframe : pandas dataframe
                        
        """

        # ENDES.1) Health Module:

        panel_health = self.obtain_health_panel_data(year_list)

        panel_health = panel_health.rename(columns={'SH227':'chlorination',
                                                    'HV201':'drinking_water_source',
                                                    'HV202':'non_drinking_water_source'}
                                                    )

        panel_health = panel_health[['year','HHID','chlorination','drinking_water_source','non_drinking_water_source']]


        # ENDES.2) Household Characteristics Module:

        panel_household = self.obtain_household_panel_data(year_list)

        panel_household = panel_household.rename(columns={'HV009':'n_members',
                                                            'HV014':'n_under_5',
                                                            'HV024':'region',
                                                            'HV025':'area',
                                                            'HV026':'area_place',
                                                            'HV040':'altitud'}
                                                            )

        panel_household = panel_household[['year','HHID','n_members','n_under_5','region','area','area_place','altitud','longitud_ccpp', 'latitud_ccpp']]

        # ENDES.3) Maternity Module:

        panel_maternity = self.obtain_maternity_panel_data(year_list)

        panel_maternity = panel_maternity.rename(columns={'B1':'month_birth_date', 
                                                            'B2':'year_birth_date', 
                                                            'B4':'child_sex',
                                                            'B5':'child_alive',
                                                            'B6':'age_death',
                                                            'B7':'age_death_month_imputed'}
                                                            )

        panel_maternity = panel_maternity[['year','HHID','month_birth_date','year_birth_date', 'child_sex', 'child_alive', 'age_death', 'age_death_month_imputed']]

        # Variables to keep:
        merged_dataframe = (panel_household.merge(panel_health, on = ['year', 'HHID'], how = 'inner')
                                            .merge(panel_maternity, on = ['year', 'HHID'], how = 'inner')
                                            )
        
        # Obtain district by plugging lat lon:
        
        return merged_dataframe





    def generate_variables_health_and_santitation(self, panel_health):

        panel_health['fuente_agua_home'] = panel_health['HV201'].isin(['Piped into dwelling',
                                                                'Dentro de la vivienda',
                                                                'Red dentro de vivienda'])

        panel_health['fuente_agua_building'] = panel_health['HV201'].isin(['Piped into dwelling',
                                                                        'Dentro de la vivienda',
                                                                        'Red dentro de vivienda',
                                                                        'Piped outside dwelling but within buikding',
                                                                        'Red fuera de la vivienda pero dentro de la edificación',
                                                                        'Fuera de la vivienda, pero dentro del edificio'])


        return panel_health


    def consolidate_variables(self, df, var_list, new_var):
        """
        Function to consolidate different variables (columns) in a DataFrame into a new variable. 
        The new variable will contain the first non-missing value from the columns in var_list for each row.
        
        Parameters:
        df (DataFrame): DataFrame to modify.
        var_list (list of str): Names of the columns to consolidate.
        new_var (str): Name of the new column to create.
        
        Returns:
        DataFrame: Modified DataFrame with the new variable added.
        """
        # Using `combine_first` to fill in missing values in the first column with values from the subsequent columns
        df[new_var] = np.nan
        for var in var_list:
            df[new_var] = df[new_var].combine_first(df[var])
        
        return df