# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 10:57:20 2021

@author: Avinash
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import pylab
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

Telecom = pd.read_excel("C:\\Users\\Avinash\\Desktop\\Assignments\\Hierarchical_clustering\\Telco_customer_churn.xlsx", sheet_name = 'Telco_Churn')

'''
Telecom.columns #To display column names
Telecom.columns.values[0] #To display column name based on index
Telecom.info()
Telecom.shape
'''
#To check particular element count in the column
#Telecom.Total_Refunds.value_counts()
#Telecom['Total Extra Data Charges'].value_counts()

#np.var(Telecom.count) #To find the variance in the column

#To replace space with "_" in all columns in the dataset 
'''
Telecom.columns = Telecom.columns.str.replace(" ","_")
Telecom.columns = Telecom.columns.str.replace(",","")
'''

Telecom1 = Telecom[['Number of Referrals','Tenure in Months','Offer', 'Avg Monthly Long Distance Charges','Internet Type','Avg Monthly GB Download','Contract','Payment Method','Monthly Charge','Total Charges','Total Long Distance Charges','Total Revenue']]

#checking unique records for each column, to decide to retain the column or not

#total number of records is 7043

'''
df['Customer ID'].nunique() #7043 --> remove no added value

df['Count'].nunique() #1 --> remove

df['Quarter'].nunique() #1 --> remove

df['Referred a Friend'].nunique() #1 --> remove

df['Number of Referrals'].nunique() #12

#unique, frequency = np.unique(df['Number of Referrals'], return_counts = True)

df['Tenure in Months'].nunique() #72

df['Offer'].nunique() #6

#unique, frequency = np.unique(df['Offer'], return_counts = True)

df['Phone Service'].nunique() #2 --> remove

df['Avg Monthly Long Distance Charges'].nunique() #3584

df['Multiple Lines'].nunique() #2 --> remove

df['Internet Service'].nunique() #2 --> remove

df['Internet Type'].nunique() #4

#unique, frequency = np.unique(df['Internet Type'], return_counts = True)

df['Avg Monthly GB Download'].nunique() #50

df['Online Security'].nunique() #2 --> remove

df['Online Backup'].nunique() #2 --> remove

df['Device Protection Plan'].nunique() #2 --> remove

df['Premium Tech Support'].nunique() #2 --> remove

df['Streaming TV'].nunique() #2 --> remove

df['Streaming Movies'].nunique() #2 --> remove

df['Streaming Music'].nunique() #2 --> remove

df['Unlimited Data'].nunique() #2 --> remove

df['Contract'].nunique() #3

#unique, frequency = np.unique(df['Contract'], return_counts = True)

df['Paperless Billing'].nunique() #2 --> remove

df['Payment Method'].nunique() #3

df['Monthly Charge'].nunique() #1585

df['Total Charges'].nunique() #6540

df['Total Refunds'].nunique() #500

df['Total Extra Data Charges'].nunique() #16 --> about 90% of the data is 0--> remove

unique, frequency = np.unique(df['Total Extra Data Charges'], return_counts = True)

df['Total Long Distance Charges'].nunique() #6110

df['Total Revenue'].nunique() #6996

'''
#df_data=Telecom1    df=Telecom
#Copying required columns
#Telecom1 = Telecom.iloc[ : , 1:30]

#Label encoding to the categorical columns
lb_make = LabelEncoder()

df_copy = Telecom1.copy(deep=True)

#df_copy['Quarter'] = lb_make.fit_transform(df_copy['Quarter'])

#df_copy['Referred a Friend'] = lb_make.fit_transform(df_copy['Referred a Friend'])

df_copy['Offer'] = lb_make.fit_transform(df_copy['Offer'])

#df_copy['Phone Service'] = lb_make.fit_transform(df_copy['Phone Service'])

#df_copy['Multiple Lines'] = lb_make.fit_transform(df_copy['Multiple Lines'])

#df_copy['Internet Service'] = lb_make.fit_transform(df_copy['Internet Service'])

df_copy['Internet Type'] = lb_make.fit_transform(df_copy['Internet Type'])

#df_copy['Online Security'] = lb_make.fit_transform(df_copy['Online Security'])

#df_copy['Online Backup'] = lb_make.fit_transform(df_copy['Online Backup'])

#df_copy['Device Protection Plan'] = lb_make.fit_transform(df_copy['Device Protection Plan'])

#df_copy['Premium Tech Support'] = lb_make.fit_transform(df_copy['Premium Tech Support'])

#df_copy['Streaming TV'] = lb_make.fit_transform(df_copy['Streaming TV'])

#df_copy['Streaming Movies'] = lb_make.fit_transform(df_copy['Streaming Movies'])

#df_copy['Streaming Music'] = lb_make.fit_transform(df_copy['Streaming Music'])

#df_copy['Unlimited Data'] = lb_make.fit_transform(df_copy['Unlimited Data'])

df_copy['Contract'] = lb_make.fit_transform(df_copy['Contract'])

#df_copy['Paperless Billing'] = lb_make.fit_transform(df_copy['Paperless Billing'])

df_copy['Payment Method'] = lb_make.fit_transform(df_copy['Payment Method'])

df_encoded = df_copy.copy()


#Normalisation
normalize = preprocessing.MinMaxScaler()
normalize_array = normalize.fit_transform(df_encoded)
df_normal = pd.DataFrame(normalize_array,columns=list(df_encoded))
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

Telecom1['clust'] = cluster_labels # creating a new column and assigning it to new column 

#Placing clust column in the beginning
Telecom = Telecom1.iloc[:, [12,0,1,2,3,4,5,6,7,8,9,10,11]]
Telecom.head()

# Aggregate mean of each cluster
Telecom = Telecom.iloc[:, 1:].groupby(Telecom.clust).mean()
Telecom.columns


# Save as csv file
Telecom.to_excel("Telecom.xlsx", encoding = "utf-8")

import os
os.getcwd()
