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

#a=Airline["cc3_miles"].nunique() #To count unique values in the column


plt.boxplot(Airline.Balance)
'''
#------------------------------------------------------------------------------------
#To check the count of outliers in each column
Q1 = Airline.quantile(0.25)
Q3 = Airline.quantile(0.75)
IQR = Q3 - Q1
count=((Airline < (Q1 - 1.5 * IQR)) | (Airline > (Q3 + 1.5 * IQR))).sum()
count
#-------------------------------------------------------------------------------------
'''
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

############### 1. Remove (let's trimm the dataset) ################
# Trimming Technique
# let's flag the outliers in the data set
#---------------------------------------------------------------------------------------
'''IQR = Airline['Flight_miles_12mo'].quantile(0.75) - Airline['Flight_miles_12mo'].quantile(0.25)
IQR
lower_limit = Airline['Flight_miles_12mo'].quantile(0.25) - (IQR * 1.5)
lower_limit
upper_limit = df['Flight_miles_12mo'].quantile(0.75) + (IQR * 1.5)
upper_limit
outliers_df = np.where(Airline['Flight_miles_12mo'] > upper_limit, True, np.where(Airline['Flight_miles_12mo'] < lower_limit, True, False))
df_trimmed = Airline.loc[~(outliers_df), ]
Airline.shape, df_trimmed.shape

# let's explore outliers in the trimmed dataset
sns.boxplot(df_trimmed.Flight_miles_12mo);plt.title('Boxplot');plt.show()
#--------------------------------------------------------------------------------------
'''

##############################################
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
df_normal = pd.DataFrame(normalize_array,columns=list(Airline))
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

from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(df_normal) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

Airline['clust'] = cluster_labels # creating a new column and assigning it to new column 

Airline = Airline.iloc[:, [7,0,1,2,3,4,5,6]]
Airline.head()

# Aggregate mean of each cluster
Airline=Airline.iloc[:, 1:].groupby(Airline.clust).mean()


Airline.to_excel("EastWestAirlines.xlsx", encoding = "utf-8")

import os
os.getcwd()
