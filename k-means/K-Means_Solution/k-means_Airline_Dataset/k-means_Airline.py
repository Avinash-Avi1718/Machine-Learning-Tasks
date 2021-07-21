# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:25:01 2021

@author: Avinash
"""

from sklearn.cluster import	KMeans
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import pylab

df = pd.read_excel("C:\\Users\\Avinash\\Desktop\\Assignments\\Hierarchical_clustering\\EastWestAirlines.xlsx", sheet_name = 'data')

df.info() #To check details of the dataset

#To rename column names
Airline = df.rename(columns={'Award?':'Award'},inplace=True)
Airline = df.rename(columns={'ID#':'ID'},inplace=True)

#To drop unwanted columns
Airline = df.drop(["ID","Award", "Qual_miles","cc2_miles","cc3_miles"], axis=1)
Airline

Airline.shape # To check rows and columns
Airline.columns #To check column names of dataset
Airline.duplicated().sum() #To check the number of duplicate rows
Airline.drop_duplicates(keep=False,inplace=True)

#There are no duplicate rows present in the dataset
#-------------------------------------------------------------------------------------------
Airline.isna().sum()#To Find missing values in the columns
#There is no missing values in the dataset
Airline.isnull().sum()
#-------------------------------------------------------------------------------------------

a=Airline["cc3_miles"].nunique() #To count unique values in the column
a

plt.boxplot(Airline.Balance)

#------------------------------------------------------------------------------------
#To check the count of outliers in each column
Q1 = Airline.quantile(0.25)
Q3 = Airline.quantile(0.75)
IQR = Q3 - Q1
count=((Airline < (Q1 - 1.5 * IQR)) | (Airline > (Q3 + 1.5 * IQR))).sum()
count
#-------------------------------------------------------------------------------------

df_winsorize = Airline.copy(deep=True)
stats.mstats.winsorize(a=df_winsorize['Balance'], limits=(0, 0.07), inplace=True)
df_winsorize.boxplot(column=['Balance'])

df_winsorize = Airline.copy(deep=True)
stats.mstats.winsorize(a=df_winsorize['Bonus_miles'], limits=(0, 0.07), inplace=True)
df_winsorize.boxplot(column=['Bonus_miles'])

df_winsorize = Airline.copy(deep=True)
stats.mstats.winsorize(a=df_winsorize['Bonus_trans'], limits=(0, 0.02), inplace=True)
df_winsorize.boxplot(column=['Bonus_trans'])

df_winsorize = Airline.copy(deep=True)
stats.mstats.winsorize(a=df_winsorize['Flight_miles_12mo'], limits=(0, 0.15), inplace=True)
df_winsorize.boxplot(column=['Flight_miles_12mo'])


#####################################
#pd.get_dummies(Airline) #To create dummy variables
#############################################

'''
#Transformation
stats.probplot(df.Balance, dist="norm",plot=pylab)
stats.probplot(df.Qual_miles, dist="norm",plot=pylab)
stats.probplot(df.Bonus_miles, dist="norm",plot=pylab)
stats.probplot(df.Bonus_trans, dist="norm",plot=pylab)
stats.probplot(df.Flight_miles_12mo, dist="norm",plot=pylab)
stats.probplot(df.Days_since_enroll, dist="norm",plot=pylab)

#To convert to normal by applying suitable function
stats.probplot(np.log(df.Balance),dist="norm",plot=pylab)
stats.probplot(np.log(df.Qual_miles),dist="norm",plot=pylab)
stats.probplot(np.log(df.Bonus_miles),dist="norm",plot=pylab)
stats.probplot(np.log(df.Bonus_trans),dist="norm",plot=pylab)
stats.probplot(np.log(df.Flight_miles_12mo),dist="norm",plot=pylab)
stats.probplot(np.log(df.Days_since_enroll),dist="norm",plot=pylab)
#sqrt,exp,log,reciprocal
'''

'''
#To standardise the data
list(Airline)
standardize = preprocessing.StandardScaler()
standardize_array = standardize.fit_transform(df)
df_standard = pd.DataFrame(standardize_array, columns=list(Airline))
df_standard
'''
#To normalise the data
normalize = preprocessing.MinMaxScaler()
normalize_array = normalize.fit_transform(Airline)
df_norm = pd.DataFrame(normalize_array,columns=list(Airline))
df_norm

###### scree plot or elbow curve ############
#k-means algorithm where we predefine the number of clusters
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
Airline['clust'] = mb # creating a  new column and assigning it to new column 

Airline.head()
df_norm.head()

Airline = Airline.iloc[:,[7,0,1,2,3,4,5,6]]
Airline.head()

#Applying aggregate function mean to each column
Airline.iloc[:, 1:].groupby(Airline.clust).mean()

Airline.to_csv("Kmeans_Airline.csv", encoding = "utf-8")

import os
os.getcwd()