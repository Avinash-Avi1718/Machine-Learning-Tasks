# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:29:53 2021

@author: Avinash
"""

#Importing libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import pylab
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#Load the dataset
Auto = pd.read_csv("C:\\Users\\Avinash\\Desktop\\Assignments\\Hierarchical_clustering\\AutoInsurance.csv")

#To check duplicated rows in the dataset
Auto.duplicated().sum() #No duplicate rows

#To check missing values in the dataset
Auto.isna().sum() #No missing values 

#To check information about dataset
Auto.info()

#Copying required columns
Auto1 = Auto.iloc[ : , 2:24]

#Label encoding to the categorical columns
lb_make = LabelEncoder()

Auto_copy = Auto1.copy(deep=True)

Auto_copy['Response'] = lb_make.fit_transform(Auto_copy['Response'])
Auto_copy['Coverage'] = lb_make.fit_transform(Auto_copy['Coverage'])
Auto_copy['Education'] = lb_make.fit_transform(Auto_copy['Education'])
Auto_copy['Effective To Date'] = lb_make.fit_transform(Auto_copy['Effective To Date'])
Auto_copy['EmploymentStatus'] = lb_make.fit_transform(Auto_copy['EmploymentStatus'])
Auto_copy['Gender'] = lb_make.fit_transform(Auto_copy['Gender'])
Auto_copy['Location Code'] = lb_make.fit_transform(Auto_copy['Location Code'])
Auto_copy['Marital Status'] = lb_make.fit_transform(Auto_copy['Marital Status'])
Auto_copy['Policy Type'] = lb_make.fit_transform(Auto_copy['Policy Type'])
Auto_copy['Policy'] = lb_make.fit_transform(Auto_copy['Policy'])
Auto_copy['Renew Offer Type'] = lb_make.fit_transform(Auto_copy['Renew Offer Type'])
Auto_copy['Sales Channel'] = lb_make.fit_transform(Auto_copy['Sales Channel'])
Auto_copy['Vehicle Class'] = lb_make.fit_transform(Auto_copy['Vehicle Class'])
Auto_copy['Vehicle Size'] = lb_make.fit_transform(Auto_copy['Vehicle Size'])


Auto_encode = Auto_copy.copy()


#Normalisation
normalize = preprocessing.MinMaxScaler()
normalize_array = normalize.fit_transform(Auto_encode)
df_normal = pd.DataFrame(normalize_array,columns=list(Auto_encode))
df_normal


# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

#Euclidean distance and complete linkage
z = linkage(df_normal, method = "complete", metric = "euclidean")


# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')

sch.dendrogram(z, 

    leaf_rotation = 0,  # rotates the x axis labels

    leaf_font_size = 10 # font size for the x axis labels

)

plt.show()

# Now applying AgglomerativeClustering choosing 6 as clusters from the above dendrogram

from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 6, linkage = 'complete', affinity = "euclidean").fit(df_normal) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

Auto1['clust'] = cluster_labels # creating a new column and assigning it to new column 

#Placing clust column in the beginning
Auto = Auto1.iloc[:, [22,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
Auto.head()

# Aggregate mean of each cluster
Auto = Auto.iloc[:, 1:].groupby(Auto.clust).mean()
Auto.columns


# Save as csv file
Auto.to_excel("Auto_Insurance.xlsx", encoding = "utf-8")

import os
os.getcwd()


