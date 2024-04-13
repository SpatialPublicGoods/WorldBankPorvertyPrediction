# Libraries
#--------------
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from joblib import load
from consolidate_ml_dataframe import DataPreparationForML


class PostEstimationRoutines(DataPreparationForML):

    def __init__(self):

        # Inherit from the parent class
        super().__init__()

        self.g_2016 = 2.4
        self.g_2017 =	0.7
        self.g_2018 =	2.0
        self.g_2019 =	0.4
        self.g_2020 =	-12.2
        self.g_2021 =	12.0
        self.growth_scale = lambda x: 1 + x/100

        self.growth_rate = {2017: np.cumprod([self.growth_scale(gg) for gg in [self.g_2017]])[-1], 
                            2018: np.cumprod([self.growth_scale(gg) for gg in [self.g_2017, self.g_2018]])[-1], 
                            2019: np.cumprod([self.growth_scale(gg) for gg in [self.g_2017, self.g_2018, self.g_2019]])[-1],
                            2020: np.cumprod([self.growth_scale(gg) for gg in [self.g_2017, self.g_2018, self.g_2019, self.g_2020]])[-1],
                            2021: np.cumprod([self.growth_scale(gg) for gg in [self.g_2017, self.g_2018, self.g_2019, self.g_2020, self.g_2021]])[-1]
                            }


    def get_variables_for_gb_model(self, lasso_model, X):

        """
        This function selects the variables to use in the gradient boosting model.
        Parameters:
            lasso_model (object): The input Lasso model object.
            X (DataFrame): The input DataFrame with the features.
        Returns:
            X: The input DataFrame with the selected features.
        """

        X = X[X.columns[lasso_model.coef_ !=0]]
        X['const'] = 1

        return X


    def generate_categorical_variables_for_analysis(self, ml_dataset):

        """
        Generate categorical variables for analysis.
        Parameters:
        ml_dataset (DataFrame): The input DataFrame.
        Returns:
        DataFrame: The input DataFrame with the added categorical variables.
        """

        ml_dataset['urbano'] = ml_dataset['strata'].isin([1,2,3,4,5]).astype(int)

        ml_dataset['trend'] = ml_dataset['year'].astype(int) - 2011

        ml_dataset['ubigeo_region'] = ml_dataset['ubigeo'].str[:4]

        ml_dataset['ubigeo_provincia'] = ml_dataset['ubigeo'].str[:6]

        ml_dataset['lima_metropolitana'] = ml_dataset['ubigeo_provincia'] == 'U-1501'

        # Get Education cateogorical variable:
        ml_dataset['educ'] = np.nan
        ml_dataset['neduc'] = 1 - (ml_dataset['prii'] + ml_dataset['seci'] + ml_dataset['secc'] + ml_dataset['supi'] + ml_dataset['supc'])
        ml_dataset.loc[ml_dataset['neduc'] == 1, 'educ'] = 'No Education'
        ml_dataset.loc[ml_dataset['prii'] == 1, 'educ'] = 'Elementary'
        ml_dataset.loc[ml_dataset['seci'] == 1, 'educ'] = 'Elementary'
        ml_dataset.loc[ml_dataset['secc'] == 1, 'educ'] = 'Elementary'
        ml_dataset.loc[ml_dataset['supi'] == 1, 'educ'] = 'Superior'
        ml_dataset.loc[ml_dataset['supc'] == 1, 'educ'] = 'Superior'

        # Get Number of children categorical variable:
        ml_dataset['n_children'] = np.nan
        ml_dataset.loc[ml_dataset['nro_hijos'] == 0, 'n_children'] = '0'
        ml_dataset.loc[ml_dataset['nro_hijos'] == 1, 'n_children'] = '1'
        ml_dataset.loc[ml_dataset['nro_hijos'] == 2, 'n_children'] = '2'
        ml_dataset.loc[ml_dataset['nro_hijos'] >= 3, 'n_children'] = '3 more'


        ml_dataset = self.input_missing_values(ml_dataset)

        return ml_dataset


    def add_random_shocks_by_region(self, ml_df, ml_df_train, error_col, region_col, shock_col, ubigeo_col):
        """
        Add a column of random shocks stratified by region to the DataFrame.
        Parameters:
        ml_df (DataFrame): The input DataFrame.
        income_col (str): The name of the column with predicted income values.
        region_col (str): The name of the column to store the region codes.
        shock_col (str): The name of the new column to store the random shocks.
        ubigeo_col (str): The name of the column containing ubigeo codes.
        Returns:
        DataFrame: The input DataFrame with the added column of random shocks.
        """
        # Copy the DataFrame to avoid modifying the original one
        df = ml_df.copy()
        # Extract region from ubigeo and create a new column for region
        df[region_col] = df[ubigeo_col].str[:4]
        # Initialize the random shock column with NaNs
        df[shock_col] = np.nan

        # Do the same for the train data so we can back out the std dev of predicted income in the region
        df_train = ml_df_train.query('year == 2016').copy()
        df_train[region_col] = df_train[ubigeo_col].str[:4]
        df_train[shock_col] = np.nan

        # Now, for each unique region, calculate the random shocks
        for region in df[region_col].unique():
            # Filter to get the predicted income for the region
            predicted_error_region = df.loc[df[region_col] == region, error_col]
            predicted_error_train_region_std = df_train.loc[df_train[region_col] == region, error_col].std()
            # Calculate the random shock for this region
            region_shock = np.random.normal(
                loc=0,
                scale=predicted_error_train_region_std ,  # scale based on the std dev of predicted income in the region
                size=predicted_error_region.shape[0]
            )
            # Assign the calculated shocks back to the main DataFrame
            df.loc[df[region_col] == region, shock_col] = region_shock

        return df


    def group_variables_for_time_series(self, grouping_variables, df, frequency='yearly'):

        """
        This function groups the DataFrame by the specified variables and calculates the mean and standard deviation of the income per capita.
        Parameters:
            grouping_variables (list): The list of variables to group by.
            df (DataFrame): The input DataFrame.
            frequency (str): The frequency of the time series. It can be 'yearly' or 'quarterly'.
        Returns:
            income_series: The DataFrame with the grouped variables and the calculated mean and standard deviation of the income per capita.
        """

        df = df.copy()

        household_weight = df['n_people']/df.groupby(grouping_variables)['n_people'].transform('sum')

        df['income_pc_weighted'] = df['income_pc'] * household_weight 
        df['income_pc_hat_weighted'] = df['income_pc_hat'] * household_weight 

        income_series = (df.groupby(grouping_variables)
                                    .agg({
                                        'income_pc_weighted': 'sum', 
                                        'income_pc_hat_weighted': 'sum',
                                        'n_people': 'count'
                                        })
                                    .reset_index()
                                    )

        income_series['std_mean'] = income_series['income_pc_weighted']/np.sqrt(income_series['n_people'])
        income_series['std_hat_mean'] = income_series['income_pc_hat_weighted']/np.sqrt(income_series['n_people'])

        # Convert 'year' and 'month' to a datetime

        if frequency == 'yearly':
            income_series['date'] = pd.to_datetime(income_series[['year']].assign(MONTH=1,DAY=1))
        elif frequency == 'quarterly':
            income_series['date'] = pd.to_datetime(income_series.rename(columns={'quarter':'month'})[['year','month']].assign(DAY=1))

        return income_series


    def group_porverty_rate_for_time_series(self, grouping_variables, df, frequency='yearly'):

        """
        This function groups the DataFrame by the specified variables and calculates the mean and standard deviation of the income per capita.
        Parameters:
            grouping_variables (list): The list of variables to group by.
            df (DataFrame): The input DataFrame.
            frequency (str): The frequency of the time series. It can be 'yearly' or 'quarterly'.
        Returns:
            income_series: The DataFrame with the grouped variables and the calculated mean and standard deviation of the income per capita.
        """

        df = df.copy()

        household_weight = df['n_people']/df.groupby(grouping_variables)['n_people'].transform('sum')
        
        df['poor_685'] = (df['income_pc'] <= df['lp_685usd_ppp']) * household_weight
        df['poor_365'] = (df['income_pc'] <= df['lp_365usd_ppp']) * household_weight
        df['poor_215'] = (df['income_pc'] <= df['lp_215usd_ppp']) * household_weight
        df['poor_hat_685'] = (df['income_pc_hat'] <= df['lp_685usd_ppp']) * household_weight
        df['poor_hat_365'] = (df['income_pc_hat'] <= df['lp_365usd_ppp']) * household_weight
        df['poor_hat_215'] = (df['income_pc_hat'] <= df['lp_215usd_ppp']) * household_weight

        income_series = (df.groupby(grouping_variables)
                                    .agg({
                                        'poor_685': 'sum', 
                                        'poor_365': 'sum',
                                        'poor_215': 'sum',
                                        'poor_hat_685': 'sum', 
                                        'poor_hat_365': 'sum',
                                        'poor_hat_215': 'sum',
                                        'n_people': 'sum'
                                        })
                                    .reset_index()
                                    )
        income_series['std_685_mean'] = np.sqrt(income_series['poor_685']*(1-income_series['poor_685']))/np.sqrt(income_series['n_people'])
        income_series['std_365_mean'] = np.sqrt(income_series['poor_365']*(1-income_series['poor_365']))/np.sqrt(income_series['n_people'])
        income_series['std_215_mean'] = np.sqrt(income_series['poor_215']*(1-income_series['poor_215']))/np.sqrt(income_series['n_people'])
        # Convert 'year' and 'month' to a datetime

        if frequency == 'yearly':
            income_series['date'] = pd.to_datetime(income_series[['year']].assign(MONTH=1,DAY=1))
        elif frequency == 'quarterly':
            income_series['date'] = pd.to_datetime(income_series.rename(columns={'quarter':'month'})[['year','month']].assign(DAY=1))

        return income_series


    def add_predicted_income_to_dataframe(self, ml_dataset, X_standardized, Y_standardized, scaler_Y, model):
        
        """
        This function adds predicted income to the DataFrame.
        Parameters:
            ml_dataset (DataFrame): The input DataFrame.
            X_standardized (DataFrame): The input DataFrame with standardized features.
            Y_standardized (DataFrame): The input DataFrame with standardized dependent variable.
            scaler_Y (StandardScaler): The input StandardScaler object for the dependent variable.
            model (object): The input model object.
        Returns:
            DataFrame: The input DataFrame with the added predicted income.
        """

        ml_dataset= ml_dataset.copy()

        # Validation:
        #------------
        ml_dataset['predicted_income'] = model.predict(X_standardized)* scaler_Y.scale_[0] + scaler_Y.mean_[0]
        ml_dataset['true_income'] = np.array(Y_standardized)* scaler_Y.scale_[0] + scaler_Y.mean_[0]
        ml_dataset['predicted_error'] = ml_dataset['predicted_income'] - ml_dataset['true_income']

        return ml_dataset




    def add_shocks_and_compute_income(self, ml_dataset, ml_dataset_train):

        """
        This function adds predicted income to the DataFrame.
        Parameters:
            ml_dataset (DataFrame): The input DataFrame.
            ml_dataset_train (DataFrame): The input DataFrame used for training the model.
            X_standardized (DataFrame): The input DataFrame with standardized features.
            Y_standardized (DataFrame): The input DataFrame with standardized dependent variable.
            scaler_Y (StandardScaler): The input StandardScaler object for the dependent variable.
            model (object): The input model object.
        Returns:
            DataFrame: The input DataFrame with the added predicted income.
        """

        # Copy the DataFrame to avoid modifying the original one
        ml_dataset= ml_dataset.copy()

        # Generate random shocks:
        random_shock_validation = np.array(self.add_random_shocks_by_region(
                                            ml_df=ml_dataset, 
                                            ml_df_train=ml_dataset_train,
                                            error_col='predicted_error', 
                                            region_col='region', 
                                            shock_col='random_shock', 
                                            ubigeo_col='ubigeo'
                                            ).random_shock
                                            )

        # Add predicted income to the DataFrame:
        ml_dataset['log_income_pc_hat'] = ml_dataset['predicted_income'] + random_shock_validation
        ml_dataset['income_pc_hat'] = np.exp(ml_dataset['log_income_pc_hat']  ) 

        return ml_dataset
    

    
    def compute_predicted_income_world_bank(self, ml_dataset):

        """
        This function adds predicted income to the DataFrame according to WB.
        Parameters:
            ml_dataset (DataFrame): The input DataFrame.
        Returns:
            DataFrame: The input DataFrame with the added predicted income.
        """

        ml_dataset['income_pc_hat'] = ml_dataset['income_pc'] * ml_dataset['year'].map(self.growth_rate)

        return ml_dataset


