
#--------------
# Libraries
#--------------
import os
import pandas as pd
from unidecode import unidecode
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from fuzzywuzzy import fuzz, process
from modules.utils_general import utils_general

#--------------
# Paths
#--------------
main_path = 'J:/My Drive/PovertyPredictionRealTime/data'
o1_path = os.path.join(main_path, '1_raw/peru/big_data/admin/')
d1_path = os.path.join(main_path, '2_intermediate')

#--------------
# Parameters
#--------------
freq = 'm'

file    = os.path.join(o1_path, 'Gasto_allyears_ubigeos/Ingreso_allyears_ubigeos.csv')

public_income = pd.read_csv(file, encoding='iso-8859-1', on_bad_lines='skip')

public_income = public_income.dropna(subset='ano_doc')

public_income = public_income.query("nivel_gobierno_nombre == 'GOBIERNOS LOCALES'")


dict_income_type = {'CANON Y SOBRECANON, REGALIAS, RENTA DE ADUANAS Y PARTICIPACIONES':'canon',
'RECURSOS DIRECTAMENTE RECAUDADOS':'recursos_directamente_recaudados',
'FONDO DE COMPENSACION MUNICIPAL':'foncomun',
'IMPUESTOS MUNICIPALES':'impuestos_municipales'}

dict_income_type.keys()

public_income = public_income.loc[public_income.rubro_nombre.isin(dict_income_type.keys())]

public_income.mes_doc.value_counts()

public_income['rubro_nombre'] = public_income['rubro_nombre'].apply(lambda x: dict_income_type[x])

public_income.rename(columns={'ano_doc':'year',
                                   'mes_doc':'month',
                                   'ubigeo_inei':'ubigeo',
                                   'monto_recaudado':'monto'
                                   }, inplace=True)

public_income = public_income.loc[:,['year','month','ubigeo','rubro_nombre','monto']]

public_income = public_income.pivot_table(index=['year','month','ubigeo'], columns='rubro_nombre', values='monto', aggfunc='sum').reset_index()

public_income = public_income.groupby(['year','month','ubigeo']).sum().reset_index()

public_income.to_csv(os.path.join(d1_path, 'public_income.csv'), index=False)

