import pandas as pd
import numpy as np
#from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("C:\\Users\\Avinash\\Desktop\\DataScience_AI\\Solutions\\Decision_Tree\\Diabetes.csv")

data.isnull().sum()
data.dropna()
data.columns
#To remove space in the beginning of the column
data.columns = data.columns.str.lstrip()

#To replace whitespace with '_' in the column names
data.columns = data.columns.str.replace(' ', '_')
data.info()


data["Class_variable"].unique()
data["Class_variable"].value_counts()
colnames = list(data.columns)

predictors = colnames[0:8]
target = colnames[8]

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