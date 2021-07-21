import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("C:\\Users\\Avinash\\Desktop\\DataScience_AI\\Solutions\\Decision_Tree\\HR_DT.csv")

data.isnull().sum()
#data.dropna()
data.columns
#To remove space in the beginning of the column
data.columns = data.columns.str.lstrip()
#Renaming columns
data.columns = data.columns.str.replace(' ', '_')
data.info()
#data = data.drop(["phone"], axis = 1)

data['target']=np.where((data['monthly_income_of_employee']>=70000) & (data['no_of_Years_of_Experience_of_employee'] <= 5), 'fake', 'genuine')

#converting into binary
lb = LabelEncoder()
data["Position_of_the_employee"] = lb.fit_transform(data["Position_of_the_employee"])

#data["default"]=lb.fit_transform(data["default"])

#data['Sales'].describe()

#data_copy = data.copy(deep = True)

data['target'].unique()
data['target'].value_counts()
colnames = list(data.columns)

predictors = colnames[0:2]
target = colnames[3]

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
