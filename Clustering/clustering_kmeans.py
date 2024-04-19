
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

final = pd.read_csv (r'D:\Establishing_flood_thresholds_for_SLR_impact_communication\Clustering\HTF_with_features_for_training_2020_45_5.csv')


final = final.drop(columns=['HTF_threshold', 'Elevation'])  
X=final.iloc[:,:]
mms = MinMaxScaler()
mms.fit(X)
data_transformed = mms.transform(X)


Sum_of_squared_distances = []
K = range(1,6)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)
    
    
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 7
plt.figure(figsize=(4, 3))
plt.plot(K, Sum_of_squared_distances, '--')
plt.xticks(K, K, fontsize=7)
plt.yticks(fontsize=7)
plt.xlabel('Clusters', fontsize=7)
plt.ylabel('Sum of Squared Distances', fontsize=7)
plt.title('Elbow Method For Optimal Clusters', fontsize=7)
plt.show()


kmeans = KMeans(n_clusters=3)
final['clusters_variables'] = kmeans.fit_predict(X)
final.to_csv(r'D:\Establishing_flood_thresholds_for_SLR_impact_communication\Clustering\Clusters.csv', index = False)
