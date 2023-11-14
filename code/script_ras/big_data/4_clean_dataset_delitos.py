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

# %% Categorize and obtain long and wide datasets:

final_denuncias_dataset = pd.read_csv(os.path.join(dataPath, '2_intermediate', 'delitos_distrito_comisaria_clean_final.csv'), index_col=False)

classification_dict = {
    "Theft and Robbery-Related Crimes": [
        'HURTO', 'ROBO', 'USURPACION', 'APROPIACION ILICITA', 'ABIGEATO', 'RECEPTACION', 'EXTORSION', 'ROBO (EN GRADO TENTATIVA)'
    ],
    "Violence and Homicide": [
        'LESIONES', 'HOMICIDIO', 'TENTATIVA DE HOMICIDIO', 'LESIONES GRAVES', 'HECHOS SEGUIDOS DE MUERTE', 'HOMICIDIO CULPOSO', 'INSTIGACION Y/O AYUDA AL SUICIDIO'
    ],
    "Sexual Offenses": [
        'VIOLACION DE LA LIBERTAD SEXUAL', 'ATENTADOS CONTRA LA PATRIA POTESTAD', 'VIOLACION DE LA LIBERTAD SEXUAL (EN GRADO TENTATIVA)', 'OFENSAS AL PUDOR PUBLICO', 'PROXENETISMO'
    ],
    "Personal Liberty Violations": [
        'VIOLACION DE LA LIBERTAD PERSONAL', 'VIOLACION DE LA LIBERTAD PERSONAL (EN GRADO TENTATIVA)', 'OTROS DELITOS CONTRA LA LIBERTAD', 'VIOLACION DE LA LIBERTAD DE EXPRESION', 'VIOLACION DEL SECRETO DE LAS COMUNICACIONES'
    ],
    "Public Safety and Health": [
        'PELIGRO COMUN', 'SALUD PUBLICA', 'EXPOSICION AL PELIGRO O ABANDONO DE PERSONA EN PELIGRO', 'DELITOS CONTRA LOS RECURSOS NATURALES', 'DELITOS DE CONTAMINACION', 'CONTRA EL ORDEN MIGRATORIO', 'RECURSOS NATURALES Y EL MEDIO AMBIENTE', 'OTROS DELITOS CONTRA LA ECOLOGIA'
    ],
    "Fraud and Financial Crimes": [
        'ESTAFA Y OTRAS DEFRAUDACIONES', 'FALSIFICACION DE DOCUMENTOS EN GENERAL', 'DELITO MONETARIO', 'DELITOS FINANCIEROS', 'LEY DE LOS DELITOS ADUANEROS (LEY 26461 DEL 08/06/95)', 'FALSIFICACION DE SELLOS, TIMBRES Y MARCAS OFICIALES', 'LIBRAMIENTOS INDEBIDOS', 'USURA', 'FRAUDE EN LA ADMINISTRACION DE PERSONAS JURIDICAS', 'OTROS DELITOS CONTRA EL ORDEN FINANCIERO Y MONETARIO', 'DEFRAUDACION FISCAL', 'LEY PENAL TRIBUTARIA (DEG. 813 20/04/96)', 'DELITOS TRIBUTARIOS PRUEBA'
    ],
    "Property and Real Estate Crimes": [
        'DAÑOS', 'VIOLACION DE DOMICILIO', 'BIENES CULTURALES', 'PROPIEDAD INDUSTRIAL', 'OTROS DELITOS CONTRA EL PATRIMONIO'
    ],
    "Public Administration Offenses": [
        'COMETIDOS POR FUNCIONARIOS PUBLICOS', 'ADMINISTRACION DE JUSTICIA', 'DISPOSICION COMUN', 'DISPOSICIONES COMUNES', 'LEY DE REPRESION DE TID (DECRETO LEY 22095)', 'MEDIOS DE TRANSPORTE, COMUNICACIONES Y OTROS SERVICIOS PUBLICOS', 'DERECHOS DE AUTOR Y CONEXOS', 'OTROS DELITOS CONTRA LA ADMINISTRACION PUBLICA', 'RESPONSABILIDAD FUNCIONAL E INFORMACION FALSA'
    ],
    "Family and Domestic Issues": [
        'OMISION DE ASISTENCIA FAMILIAR', 'OTROS DELITOS CONTRA LA VIDA, EL CUERPO Y LA SALUD', 'OTROS DELITOS CONTRA LA FAMILIA', 'MATRIMONIOS ILEGALES'
    ],
    "Public Order and Political Crimes": [
        'NO INDICA', 'DISCRIMINACION', 'CONTRA LA HUMANIDAD', 'LEY ORGANICA DE ELECCIONES (LEY 26859 01/10/97)', 'VIOLACION DE LA LIBERTAD DE TRABAJO', 'DERECHO DE SUFRAGIO', 'PENALIDAD PARA DELITOS DE TERRORISMO (D.LEY 25475 06/05/92)', 'PAZ PUBLICA', 'OTROS DELITOS CONTRA LA SEGURIDAD PUBLICA', 'OTROS DELITOS CONTRA LA TRANQUILIDAD PUBLICA', 'OTROS DELITOS CONTRA LA FE PUBLICA', 'REBELION, SEDICION Y MOTIN', 'ATENTADOS CONTRA LA SEGURIDAD NACIONAL Y TRAICION A LA PATRIA', 'DELITOS QUE COMPROMETEN LAS RELACIONES EXTERIORES DEL ESTADO', 'OTROS DELITOS CONTRA LOS PODERES DEL ESTADO Y EL ORDEN CONSTITUCIONAL', 'OTROS DELITOS CONTRA EL ESTADO Y LA DEFENSA NACIONAL'
    ],
    "Information and Cyber Crimes": [
        'DELITOS INFORMATICOS', 'INJURIA, CALUMNIA Y DIFAMACION', 'VIOLACION DEL SECRETO PROFESIONAL', 'OTROS DELITOS CONTRA LA LIBERTAD DE EXPRESION', 'OTROS DELITOS CONTRA EL HONOR'
    ],
    "Economic and Commercial Offenses": [
        'OTROS DELITOS ECONOMICOS', 'ACAPARAMIENTO, ESPECULACION Y ADULTERACION', 'OTROS DELITOS CONTRA LA CONFIANZA Y LA BUENA FE EN LOS NEGOCIOS', 'OTROS DELITOS CONTRA EL ORDEN ECONOMICO', 'ABUSO DEL PODER ECONOMICO', 'ELABORACION Y COMERCIO CLANDESTINO DE PRODUCTOS', 'VENTA ILICITA DE MERCADERIAS', 'CONTRABANDO'
    ],
    "Intellectual Property and Cultural Heritage": [
        'DERECHOS DE AUTOR Y CONEXOS', 'OTROS DELITOS CONTRA LOS DERECHOS INTELECTUALES', 'PROPIEDAD INDUSTRIAL'
    ],
    "Miscellaneous Offenses": [
        'OTROS', 'OTRO', 'OTROS DELITOS TRIBUTARIOS', 'GENOCIDIO', 'QUIEBRA', 'OTROS DELITOS CONTRA LA HUMANIDAD', 'INJURIA', 'CALUMNIA', 'VIOLACION DE LA LIBERTAD DE REUNION', 'DELITOS CONTRA EL ESTADO CIVIL', 'ATENTADOS CONTRA EL SISTEMA CREDITICIO', 'DELITOS CONTRA LOS SIMBOLOS Y VALORES DE LA PATRIA', 'DIFAMACION'
    ]
}

inverted_dict = {v: k for k, values in classification_dict.items() for v in values}

final_denuncias_dataset['especifica_aggregated'] = final_denuncias_dataset['Especifica'].map(inverted_dict)


final_denuncias_dataset_long = pd.melt(final_denuncias_dataset, 
                                       id_vars=['Distrito de la denuncia', 'Comisaria', 'Especifica', 'Período', 'ubigeo', 'distrito', 'comisaria_code', 'comisaria_name', 'especifica_aggregated'],
                                       value_vars=['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'],
                                       var_name='Year', value_name='Cases')

final_denuncias_dataset_long['Cases'] = final_denuncias_dataset_long['Cases'].fillna(0)

final_denuncias_dataset_long['Cases'] = final_denuncias_dataset_long['Cases'].str.replace(' ', '')

final_denuncias_dataset_long['Cases'] = pd.to_numeric(final_denuncias_dataset_long['Cases'], errors='coerce')

final_denuncias_dataset_long['Cases'] = final_denuncias_dataset_long['Cases'].fillna(0)

# Now get columns for each case:

final_denuncias_dataset_wide = final_denuncias_dataset_long.pivot_table(index=['Year', 'Distrito de la denuncia', 'Comisaria', 'ubigeo', 'distrito', 'comisaria_code', 'comisaria_name'],
                                                    columns='especifica_aggregated', 
                                                    values='Cases',
                                                    aggfunc='sum',  # Change to 'count' or another function if needed
                                                    fill_value=0)   # Replaces NaN with 0 for non-existing cases

# Reset index if you want 'Year', 'Distrito de la denuncia', etc., as columns

final_denuncias_dataset_wide = final_denuncias_dataset_wide.reset_index()

column_name_mapping = {
    'Economic and Commercial Offenses': 'Economic_Commercial_Offenses',
    'Family and Domestic Issues': 'Family_Domestic_Issues',
    'Fraud and Financial Crimes': 'Fraud_Financial_Crimes',
    'Information and Cyber Crimes': 'Information_Cyber_Crimes',
    'Intellectual Property and Cultural Heritage': 'Intellectual_Property_Cultural_Heritage',
    'Miscellaneous Offenses': 'Miscellaneous_Offenses',
    'Personal Liberty Violations': 'Personal_Liberty_Violations',
    'Property and Real Estate Crimes': 'Property_Real_Estate_Crimes',
    'Public Administration Offenses': 'Public_Administration_Offenses',
    'Public Order and Political Crimes': 'Public_Order_Political_Crimes',
    'Public Safety and Health': 'Public_Safety_Health',
    'Sexual Offenses': 'Sexual_Offenses',
    'Theft and Robbery-Related Crimes': 'Theft_Robbery_Related_Crimes',
    'Violence and Homicide': 'Violence_Homicide'
}

# Assuming 'pivot_df' is your DataFrame
final_denuncias_dataset_wide = final_denuncias_dataset_wide.rename(columns=column_name_mapping)


final_denuncias_dataset_wide.to_csv(os.path.join(dataPath, '3_working', 'delitos_distrito_comisaria_clean_panel.csv'), index=False)

