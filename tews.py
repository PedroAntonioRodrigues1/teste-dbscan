from sklearn.cluster import DBSCAN
from haversine import haversine, Unit
import numpy as np
import pandas as pd


df = pd.read_csv('tsx.csv', header = None, sep=';', decimal=',', skiprows=1)


coordenadas = np.radians(df[[1, 2]].to_numpy())


kms_per_radian = 6371.0088
epsilon = 1.5 / kms_per_radian

# DBSCAN
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(coordenadas)

print(db.labels_[:5])

