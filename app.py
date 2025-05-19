# %%
#Cargamos librerías principales

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
#Cargamos el modelo

import pickle
filename = 'modelo-reg-tree-knn-nn.pkl'
#filename = 'modelo-reg-tree.pkl'
model_Tree,model_Knn, model_NN,variables, min_max_scaler = pickle.load(open(filename, 'rb')) #DT-Knn
#model_Tree,variables = pickle.load(open(filename, 'rb'))# DT
#model_Tree,variables = pickle.load(open(filename, 'rb')) #DT-Knn

# %%
#Cargamos los datos futuros

data = pd.read_csv("videojuegos-datosFuturos.csv", sep = ",")
data.head()

# %%
#Se realiza la preparación

data_preparada=data.copy()
data_preparada = pd.get_dummies(data_preparada, columns=['videojuego', 'Plataforma'], drop_first=False)
data_preparada = pd.get_dummies(data_preparada, columns=['Sexo', 'Consumidor_habitual'], drop_first=True)# se elimina una dummy porque solo tiene 2 categorias
data_preparada.head()

# %%
#Se adicionan las columnas faltantes

data_preparada=data_preparada.reindex(columns=variables,fill_value=0)# Si falta una variable la crea y llena con ceros
data_preparada.head()

# %%
#Hacemos la predicción

Y_Tree = model_Tree.predict(data_preparada)

print(Y_Tree)

# %%
data['Prediccion_Tree']=Y_Tree

data

# %%
#Se normaliza la edad
data_preparada[['Edad']]= min_max_scaler.transform(data_preparada[['Edad']])
data_preparada

# %%
#Hacemos la predicción
Y_Knn = model_Knn.predict(data_preparada)
data['Prediccion_Knn']=Y_Knn
data

# %%
#Hacemos la predicción

Y_NN = model_NN.predict(data_preparada)
data['Prediccion_NN']=Y_NN
data


