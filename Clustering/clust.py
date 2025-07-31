

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


data = pd.read_csv("DataSet/bank-full.csv")

# print(f"Data shape is {data.shape[0]} rows and {data.shape[1]} columns")

# print(f"check nulls in data {data.isna().sum()}")

# print(f"Data info is {data.info()}")
# print(f"Data colmuns are is {data.columns}")


Object_Columns = ['job', 'marital', 'education', 'default','housing', 'loan',
                 'contact', 'month', 'day_of_week','poutcome', 'subscribed']

print(f"Data of object columns are \n {data[Object_Columns]}")


for i in Object_Columns:
    Dict = {i:j for i,j in zip(data[i].unique(),range(data[i].nunique()))}
    data[f"Enc_{i}"] = data[i].map(Dict)
    data.drop([i],axis = 1, inplace = True)



print(f"Data after conversion is \n{data}")

kmeans_Model = KMeans(n_clusters= 5 , init = 'k-means++' , random_state= 33 , algorithm='lloyd') 

kmeans_Model.fit(data)

print(pd.DataFrame(kmeans_Model.cluster_centers_, columns=data.columns,index=['Cluster A','Cluster B','Cluster C','Cluster D','Cluster E']))

print(pd.Series(kmeans_Model.labels_).value_counts())

print(kmeans_Model.inertia_)

dict_interia = {}
for i in range(2,16):
    kmeans_Model = KMeans(n_clusters= i , init = 'k-means++' , random_state= 33 , algorithm='lloyd') 
    kmeans_Model.fit(data)
    dict_interia[i] = kmeans_Model.inertia_

print(dict_interia)

plt.figure(figsize=(14,7))
plt.title("elbow Method")

plt.xlabel("No of Clusters")
plt.ylabel("Intertia / SSD")

plt.plot(dict_interia.keys(), dict_interia.values(), 
         color='red', marker='o', linestyle='dashed', linewidth=2, markersize=8)
plt.show()