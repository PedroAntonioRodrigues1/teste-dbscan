from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import numpy as np
import pandas as pd



def geodesic_distance(X, Y):
    return geodesic((X[0], X[1]), (Y[0], Y[1])).km

df = pd.read_csv('tsx.csv', header = 0, sep=';', decimal=',' , dtype={'cep': float , 'latitude': float , 'longitude': float} ,
                 )

df.drop(df[(df['latitude'] < -90) | (df['latitude'] > 90)].index, inplace=True)
df.drop(df[(df['longitude'] < -180) | (df['longitude'] > 180)].index, inplace=True)
coordenadas = np.radians(df[['latitude', 'longitude']].to_numpy())


# Parâmetros
kms_per_radian = 6371.0088
epsilon = 1.5 / kms_per_radian

# DBSCAN
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric=geodesic_distance).fit((coordenadas))


print(db.labels_[:5])

