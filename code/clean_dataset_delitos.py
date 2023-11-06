import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules.utils_general import utils_general as ug

dataPath = 'J:/My Drive/PovertyPredictionRealTime/data'

# This dataset contains the number of crimes committed in each municipality of Peru.

crime = pd.read_excel(os.path.join(dataPath,'1_raw','peru','big_data','admin', 'delitos', 'delitos_distrito_comisaria_to_fix.xlsx'))

crime_compiler = crime.copy()

mask = crime['Distrito de la denuncia'].str.contains(r'^\d{5,}', regex=True)

# Get the indices of rows with numbers of more than 4 digits
indices_with_large_numbers = crime.index[mask].tolist()

mask = crime['Distrito de la denuncia'].str.contains(r'^\d{3,4}|\bCOMISARIA\b', regex=True)

# Get the indices of rows with numbers of 3 to 4 digits
indices_with_3_to_4_digits = crime.index[mask].tolist()


indices_with_large_numbers.sort()
indices_with_3_to_4_digits.sort()

indices_between_dict = {}

for ii in range(len(indices_with_large_numbers)):

    if ii < len(indices_with_large_numbers)-1:
        start_index = indices_with_large_numbers[ii] 
        end_index = indices_with_large_numbers[ii+1]

        between_indices = [idx for idx in indices_with_3_to_4_digits if start_index <= idx <= end_index]
        
        # Include indices at the beginning and end of the list
        indices_between_dict[start_index] = between_indices 

    else:
        start_index = indices_with_large_numbers[ii] 
        end_index = crime.shape[0]

        between_indices = [idx for idx in indices_with_3_to_4_digits if start_index <= idx <= end_index]
        
        # Include indices at the beginning and end of the list
        indices_between_dict[start_index] = between_indices + [end_index]


for jj in indices_between_dict.keys():

    list_test = indices_between_dict[jj] 

    for ii in range(len(list_test)-1):

        if ii == 0:
            
            # Move all crime category rows
            crime_compiler.iloc[range(list_test[ii]+1,list_test[ii+1]),2:17] = crime.iloc[range(list_test[ii]+1,list_test[ii+1]),0:15]
            crime_compiler.iloc[range(list_test[ii]+1,list_test[ii+1]),0] = np.nan

        else:
            # Move comisaria row
            crime_compiler.iloc[list_test[ii],range(1,17)] = crime.iloc[list_test[ii],range(0,16)]
            crime_compiler.iloc[list_test[ii],0] = np.nan

            # Move all crime category rows
            crime_compiler.iloc[range(list_test[ii]+1,list_test[ii+1]),2:17] = crime.iloc[range(list_test[ii]+1,list_test[ii+1]),0:15]
            crime_compiler.iloc[range(list_test[ii]+1,list_test[ii+1]),0] = np.nan


crime_compiler.to_csv(os.path.join(dataPath, '2_intermediate', 'delitos_distrito_comisaria_to_fix_clean.csv'), index=False)

# Append with crime_fixed:
crime_fixed = pd.read_excel(os.path.join(dataPath,'1_raw','peru','big_data', 'admin', 'delitos', 'delitos_distrito_comisaria_fixed.xlsx'))


final_denuncias_dataset = pd.concat([crime_fixed, crime_compiler],axis=0)

for ii in range(final_denuncias_dataset.shape[0]):
    
    if ii > 0:
        if pd.isna(final_denuncias_dataset.iloc[ii,0]):
            final_denuncias_dataset.iloc[ii,0] = final_denuncias_dataset.iloc[ii-1,0]

        if pd.isna(final_denuncias_dataset.iloc[ii,1]):
            final_denuncias_dataset.iloc[ii,1] = final_denuncias_dataset.iloc[ii-1,1]

pattern = r'(\d+) (.*)'
final_denuncias_dataset[['ubigeo', 'distrito']] = final_denuncias_dataset['Distrito de la denuncia'].str.extract(pattern)

final_denuncias_dataset[['comisaria_code', 'comisaria_name']] = final_denuncias_dataset['Comisaria'].str.extract(pattern)

final_denuncias_dataset.to_csv(os.path.join(dataPath, '2_intermediate', 'delitos_distrito_comisaria_clean_final.csv'), index=False)