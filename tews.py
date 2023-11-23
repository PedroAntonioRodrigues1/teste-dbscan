from sklearn.cluster import DBSCAN
from haversine import haversine, Unit
import numpy as np
import pandas as pd

#Função personalizada de distância
def haversine_distance(X, Y):
    return haversine((X[0], X[1]), (Y[0], Y[1]))

df = pd.read_csv('tsx.csv', header = None, sep=';', decimal=',', skiprows=1)

# Verificar se a coluna é do tipo object (string) antes de tentar substituir
if df[1].dtype == 'object':
    df[1] = df[1].str.replace(',', '.').astype(float)
if df[2].dtype == 'object':
    df[2] = df[2].str.replace(',', '.').astype(float)

# Filtrar valores de latitude e longitude fora do intervalo
df = df[(df[1] >= -90) & (df[1] <= 90)]
df = df[(df[2] >= -180) & (df[2] <= 180)]

#coordenadas = df.to_numpy()
coordenadas = np.radians(df[[1, 2]].to_numpy())

# Parâmetros
kms_per_radian = 6371.0088
epsilon = 1.5 / kms_per_radian

# DBSCAN
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric=haversine_distance).fit(coordenadas)

print(db.labels_)
