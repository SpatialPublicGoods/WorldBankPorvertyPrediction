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
from global_settings import global_settings


class GenerateFiguresReport(global_settings):

    def __init__(self):

        # Inherit from the parent class
        super().__init__()

        
    def create_binned_scatterplot(self, df, income_col, predicted_col, bin_width=0.1, bin_start=0, bin_end=1):
        # Define the bins based on specified parameters
        bins = np.arange(bin_start, bin_end + bin_width, bin_width)
        # Categorize the log income data into bins
        df['income_bin'] = pd.cut(df[income_col], bins=bins, labels=[f"{round(b, 2)}-{round(b+bin_width, 2)}" for b in bins[:-1]])
        # Calculate the mean of the predicted log income for each bin
        binned_data = df.groupby('income_bin')[predicted_col].mean().reset_index()
        binned_data['income_bin'] = binned_data['income_bin'].str.split('-').str[-1].astype(float)
        binned_data = binned_data.dropna()
        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 7))
        # Histogram
        sns.histplot(df[income_col], 
                    color=self.color6, 
                    linewidths=2,
                    label='True Income', 
                    stat='density',
                    element='step',
                    alpha=0.2,
                    ax=ax1
                    )
        ax1.set_ylabel('Density')
        # Poverty lines:
        ax1.axvline(x=np.log(208.35417), color='gray', linestyle='--', linewidth=1)
        ax1.axvline(x=np.log(111.020836), color='gray', linestyle='--', linewidth=1)
        ax1.axvline(x=np.log(65.395836), color='gray', linestyle='--', linewidth=1)
        # Binned scatterplot
        ax2 = ax1.twinx()
        ax2.scatter(binned_data['income_bin'], 
                    binned_data[predicted_col], 
                    alpha=0.8, 
                    color=self.color1,
                    zorder=5 , # Ensure scatterplot is on top
                    s=150
                    )
        ax2.set_ylabel('Average Predicted Log Income', color=self.color1)
        ax2.tick_params(axis='y', labelcolor=self.color1)
        ax2.set_xticks(binned_data['income_bin'])
        ax1.set_xticklabels([f"{round(b, 1)}" for b in binned_data['income_bin']], rotation=90, fontsize='small')
        # fig.suptitle('Binned Scatterplot of Predicted Log Income')
        fig.tight_layout()  # Adjust layout to not overlap
        plt.xlim([2 , 9.9])
        fig.savefig('../figures/fig0_binned_scatterplot.pdf', bbox_inches='tight')
        return print('Figure 0 saved')