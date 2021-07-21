# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 19:45:11 2021

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

#Load the dataset
crime = pd.read_csv("C:\\Users\\Avinash\\Desktop\\Assignments\\Hierarchical_clustering\\crime_data.csv")

#To display the columns
crime.columns
crime.info()

#To display the information
crime.info()
#crime.rename( columns={'Unnamed: 0':'Country'}, inplace=True ) #To name the unnamed column 
#In the given dataset, first column is unnamed. hence given suitable column name

#Dropping the unwanted columns which we do not need for model
crime1 = crime.drop(["Unnamed: 0"], axis=1)

#To check duplicate rows
crime1.duplicated().sum() #No duplicate rows in the dataset

#To check missing values in the dataset
crime1.isna().sum() #No missing values/NAN values in the dataset

# To count the number of outliers in the columns
Q1 = crime1.quantile(0.25)
Q3 = crime1.quantile(0.75)
IQR = Q3 - Q1
count=((crime1 < (Q1 - 1.5 * IQR)) | (crime1 > (Q3 + 1.5 * IQR))).sum()
count

#To check outliers using boxplot
plt.boxplot(crime1.Rape)

IQR = crime1['Rape'].quantile(0.75) - crime1['Rape'].quantile(0.25)
IQR
lower_limit = crime1['Rape'].quantile(0.25) - (IQR * 1.5)
lower_limit
upper_limit = crime1['Rape'].quantile(0.75) + (IQR * 1.5)
upper_limit

#To remove outliers by winsorise method
df_winsorize = crime1.copy(deep=True)
stats.mstats.winsorize(a=df_winsorize['Rape'], limits=(0, 0.04), inplace=True)
df_winsorize.boxplot(column=['Rape'])

#Normalisation
normalize = preprocessing.MinMaxScaler()
normalize_array = normalize.fit_transform(crime1)
df_normal = pd.DataFrame(normalize_array,columns=list(crime1))
df_normal


from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_normal, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Plotting agglomerative clustering by choosing 2 as number of clusters
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 2, linkage = 'complete', affinity = "euclidean").fit(df_normal) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

crime1['clust'] = cluster_labels # creating a new column and assigning it to new column 

crime = crime1.iloc[:, [4,0,1,2,3]]
crime.head()

# Aggregate mean of each cluster
crime = crime.iloc[:, 1:].groupby(crime.clust).mean()

# Save as csv file
crime.to_excel("crime.xlsx", encoding = "utf-8")

import os
os.getcwd()