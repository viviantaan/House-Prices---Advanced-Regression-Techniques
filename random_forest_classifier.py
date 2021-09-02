# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:00:00 2021

@author: Usuario
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import random


random.seed(12345)

base = pd.read_csv('classesTOG_dummy2.csv', sep=';', decimal=',')
baseFiscal = pd.read_csv('baseTeste_fiscal.csv', sep=';', decimal=',')

#caso seja necessário apagar algumas colunas, usar o comando abaixo
base = base.drop(['TOG'], axis=1)

previsores = base.iloc[:,0:base.shape[1]-1].values
classe = base.iloc[:,base.shape[1]-1].values

previsores_testeFiscal = baseFiscal.iloc[:,0:baseFiscal.shape[1]-1].values
classe_testeFiscal = baseFiscal.iloc[:,baseFiscal.shape[1]-1].values

#-1 no final para não pegar feição como o nome de uma feature
#nomes = base.columns[0:base.shape[1]-1]

#quando considerar apenas um tipo de TOG
#labelencoder_classe = LabelEncoder()
#previsores[:,39] = labelencoder_classe.fit_transform(previsores[:,39])
#classe = labelencoder_classe.fit_transform(classe)

#quando considerar os 3 tipos de TOG numa planilha
labelencoder_classe = LabelEncoder()
#previsores[:,previsores.shape[1]-3] = labelencoder_classe.fit_transform(previsores[:,previsores.shape[1]-3])
#previsores[:,previsores.shape[1]-2] = labelencoder_classe.fit_transform(previsores[:,previsores.shape[1]-2])
#previsores[:,previsores.shape[1]-1] = labelencoder_classe.fit_transform(previsores[:,previsores.shape[1]-1])
classe = labelencoder_classe.fit_transform(classe)

#quando considerar apenas um tipo de TOG
#imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose=0)
#imputer = imputer.fit(previsores[:, 0:previsores.shape[1]-1])
#previsores[:, 0:previsores.shape[1]-1] = imputer.transform(previsores[:, 0:previsores.shape[1]-1])

#quando considerar os 3 tipos de TOG numa planilha
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose=0)
imputer = imputer.fit(previsores[:, 0:previsores.shape[1]-3])
previsores[:, 0:previsores.shape[1]-3] = imputer.transform(previsores[:, 0:previsores.shape[1]-3])

#quando considerar apenas um tipo de TOG
#scaler = StandardScaler()
#previsores[:, 0:previsores.shape[1]-1] = scaler.fit_transform(previsores[:,0:previsores.shape[1]-1])

#previsores1 = pd.DataFrame(previsores)
#previsores1.to_csv(r'C:\Users\Usuario\Documents\Doutorado\Petrobras\2. RAINBOW\P-35\previsores1.csv', sep=';', decimal=',', index=False)

#quando considerar os 3 tipos de TOG numa planilha
#scaler = StandardScaler()
#previsores[:, 0:previsores.shape[1]-3] = scaler.fit_transform(previsores[:,0:previsores.shape[1]-3])
rs = 4
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.1, random_state=rs)
#apenas caso tenha que ser feito o conjunto de validação
#previsores_treinamento, previsores_val, classe_treinamento, classe_val = train_test_split(previsores_treinamento, classe_treinamento, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
#n = len(classe_val)

data_treinamento= previsores_treinamento[:,0]
data_treinamento= pd.DataFrame(data_treinamento)
data_treinamento= data_treinamento.astype(int)

data_teste= previsores_teste[:,0]
data_teste= pd.DataFrame(data_teste)
data_teste= data_teste.astype(int)

classificador_random_forest = RandomForestClassifier(n_estimators = 100, random_state=0)
classificador_random_forest.fit(previsores_treinamento, classe_treinamento)
previsoes_random_forest = classificador_random_forest.predict(previsores_testeFiscal)
precisao_random_forest = accuracy_score(classe_teste,previsoes_random_forest)
matriz_random_forest = confusion_matrix(classe_teste,previsoes_random_forest)

probabilidades_train_test = classificador_random_forest.predict_proba(previsores[:previsores.shape[0]])
probabilidades_previsoes = classificador_random_forest.predict_proba(previsores_teste[:previsores_teste.shape[0]])

print('Precisão do Random Forest: ', precisao_random_forest)

