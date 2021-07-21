import pandas as pd
#import numpy as np
# import seaborn as sb
#import matplotlib.pyplot as plt

#import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 

#Importing Data
election_data = pd.read_csv("C:\\Users\Avinash\OneDrive\Desktop\DataScience_AI\Solutions\Logistic_Regression\\election_data.csv", sep = ",")

#Preprocessing
election_data.info()
election_data.Country.value_counts()
c1 = election_data
c1.head(11)
c1.describe()
c1.isna().sum()
c1.columns
column_names = []
for i in range(len(c1.columns)):
    column_names.append(c1.columns[i].replace(' ', '_'))
c1.columns = column_names
column_names = []
for i in range(len(c1.columns)):
    column_names.append(c1.columns[i].replace('-', '_'))
c1.columns = column_names
c1.info()
c1 = c1.dropna()
#############################################################

X_train, X_test, y_train, y_test = train_test_split(c1[['Election_id','Year','Amount_Spent','Popularity_Rank']],c1[['Result']],train_size=0.8)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

model.score(X_test,y_test)