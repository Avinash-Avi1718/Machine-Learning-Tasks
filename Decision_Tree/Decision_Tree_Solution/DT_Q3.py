import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("C:\\Users\\Avinash\\Desktop\\DataScience_AI\\Solutions\\Decision_Tree\\Fraud_check.csv")
data.isnull().sum()
data.columns
data.info()

#Putting condition as given
data['Taxable.Income'] = np.where(data['Taxable.Income']<=30000, 'Risky', 'Good')

#converting into binary
lb = LabelEncoder()
data["Undergrad"] = lb.fit_transform(data["Undergrad"])
data["Marital.Status"] = lb.fit_transform(data["Marital.Status"])
data["Urban"] = lb.fit_transform(data["Urban"])

data["Taxable.Income"].unique()
data["Taxable.Income"].value_counts()
colnames = list(data.columns)

data.columns[[0,1,3,4,5]]

predictors = data.columns[[0,1,3,4,5]]
target = colnames[2]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

#help(DT)
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