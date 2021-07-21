import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("C:\\Users\\Avinash\\OneDrive\\Desktop\\DataScience_AI\\Solutions\\Decision_Tree\\Company_Data.csv")

data.isnull().sum()
data.dropna()
data.columns
data.info()
#data = data.drop(["phone"], axis = 1)
#Axis=1 means, we need to drop column but not row

#converting into binary
lb = LabelEncoder()
data["ShelveLoc"] = lb.fit_transform(data["ShelveLoc"])
data["Urban"] = lb.fit_transform(data["Urban"])
data["US"] = lb.fit_transform(data["US"])

data['Sales'].describe()

#converting continuous data to categorical data
data['Sales'] = pd.cut(data.Sales, bins=6, labels=np.arange(6), right=False)

data['Sales'].unique()
data['Sales'].value_counts()
colnames = list(data.columns)

predictors = colnames[1:11]
target = colnames[0]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT
help(DT)
model = DT(criterion = 'entropy',  max_depth=5)
model.fit(train[predictors], train[target])

from sklearn.tree import plot_tree

plot_tree(model)

# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy