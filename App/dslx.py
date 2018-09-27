#!/usr/bin/env python

"""
This script is designed to perform a prediction of dyslexia in children.
A part of Dyslexia Project led in cooperation with Nencki Institute of Experimental Biology, Warsaw, Poland.
"""

__author__ = "Aleksander Molak"
__copyright__ = "Copyright 2018, DSLX Project"
__credits__ = ["Aleksander Molak"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Aleksander Molak, Katarzyna Chyl"
__email__ = "aleksnader.molak@gmail.com"
__status__ = "Testing"

import numpy as np

hello_list = ['Jak się masz?', 'Jak Ci mija dzień?', 'Dzień dobry!', 'Cześć!', 'Hej! Jak tam u Ciebie?']

hello_phrase = hello_list[np.random.randint(5)]

print(f'\n\n{hello_phrase} :)\n\n\nWczytuję moduły...\n\n')


# Import libs
import sys
import pickle

import pandas as pd

from datetime import datetime

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# Define paths
models_path = r'Models'

# Import models and data // transform the data
with open(rf'{models_path}/scaler1.pkl', 'rb') as file:
    scaler1 = pickle.load(file)

with open(rf'{models_path}/gbc_params.pkl', 'rb') as file:
    gbc_params = pickle.load(file)

with open(rf'{models_path}/svc_params.pkl', 'rb') as file:
    svc_params = pickle.load(file)

X_gbc = pd.read_csv(rf'{models_path}/X_gbc.csv')
X_svc = pd.read_csv(rf'{models_path}/X_svc.csv')
y_gbc = pd.read_csv(rf'{models_path}/y_gbc.csv')
y_svc = pd.read_csv(rf'{models_path}/y_svc.csv')

y_gbc = y_gbc.values.ravel()
y_svc = y_svc.values.ravel()

# Initialize the classifiers
print('Inicjalizuję modele...\n\n')
gbc = GradientBoostingClassifier(**gbc_params, random_state=4)
svc = SVC(**svc_params)

# Fit the classifiers
gbc.fit(X_gbc, y_gbc)
svc.fit(X_svc, y_svc)
print('Modele zainicjalizowane!\n\n')

# Read-in the test file
file = sys.argv[1]

X_test = pd.read_csv(file)

# Check and transform the data
print('Sprawdzam strukturę danych...\n\n')

if 'Unnamed: 0' in X_test.columns:
    X_test = X_test.drop('Unnamed: 0', axis=1)

contains_ID = False

if 'ID' in X_test.columns:
    contains_ID == True
    X_test = X_test.drop('ID', axis=1)
    ID = X_test.ID

X_cols = X_test.columns

# Define expected columns
column_list = ['TURA1_ZnajomoscLiter', 'TURA1_AnalizaFonemowa',
       'TURA1_UsuwanieFonemow', 'TURA1_TSN_PrzedmiotySurowy',
       'TURA1_TSN_KolorySurowy', 'TURA1_SlownictwoSurowy',
       'TURA1_GrafotaktykaSurowy', 'TURA1_PowtarzanieCyfr',
       'TURA1_PowtarzanieSylab', 'SRD_21', 'ARHQ_M_17', 'ARHQ_M_21', 'SRD_29',
       'SRD_1', 'ARHQ_M_10', 'SRD_9']


# Check if the file contains expected columns
if sum(X_test.columns == column_list) != 16:
    raise ValueError(f'Column names in your files do not match expected values.\nYour file should contain the following columns: \n{column_list}')
else:
    print('Struktura danych OK\n\n')

# Scale the Data
print('Przekształcam dane...\n\n')
X_test_n = scaler1.fit_transform(X_test)
print('Dane przekształcone!\n\n')

# Predict probas
print('Generuję predykcje...\n\n')
gbc_proba = gbc.predict_proba(X_test_n)
svc_proba = svc.predict_proba(X_test_n)
print('Faza 1 predykcji: sukces!\n\n')

# Build an ensemble prediction
meta_proba = (gbc_proba + svc_proba) / 2
meta_pred = meta_proba[:,1] > .5
meta_pred = [1 if x==True else 0 for x in meta_pred]
print('Faza 2 predykcji: sukces!\n\n')

# Transform X_test to a data frame
# X_test = pd.DataFrame(X_test, columns=X_cols)

# Join the frames
X_test['Predykcja'] = meta_pred
if contains_ID == True:
    X_test = ID.join(X_test)

# Generate a unique filename
filename = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_DslxApp_Prediction.csv'

# Save the file
print('Zapisuję Twoje predykcje...\n\n')
X_test.to_csv(f'Predictions/{filename}', index=False)
print(f'Plik {filename} został zapisany.')

goodbye_list = ['Do zobaczenia!', 'Miłego dnia!']

goodbye_phrase = goodbye_list[np.random.randint(2)]

# Say goodbye
print(f'\n\nDzięki! {goodbye_phrase} :)\n\n')
