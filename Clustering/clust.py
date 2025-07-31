

import pandas as pd
from sklearn.cluster import KMeans


data = pd.read_csv("DataSet/bank-full.csv")

# print(f"Data shape is {data.shape[0]} rows and {data.shape[1]} columns")

# print(f"check nulls in data {data.isna().sum()}")

# print(f"Data info is {data.info()}")
# print(f"Data colmuns are is {data.columns}")


Object_Columns = ['job', 'marital', 'education', 'default','housing', 'loan',
                 'contact', 'month', 'day_of_week','poutcome', 'subscribed']

print(f"Data of object columns are \n {data[Object_Columns]}")

# print(f"data['job'].unique() {data['job'].nunique()}")
# print(f"data['job'].unique() {zip(data['job'].unique(),range(5))}")
for i in Object_Columns:
    Dict = {i:j for i,j in zip(data[i].unique(),range(data[i].nunique()))}
    data[f"Enc_{i}"] = data[i].map(Dict)
    data.drop([i],axis = 1, inplace = True)



print(f"Data after conversion is \n{data}")
kmeans_Model = KMeans(n_clusters= 5 , init = 'k-means++' , random_state= 33 , algorithm='lloyd') 

kmeans_Model.fit(data)

print(pd.DataFrame(kmeans_Model.cluster_centers_, columns=data.columns,index=['Cluster A','Cluster B','Cluster C','Cluster D','Cluster E']))