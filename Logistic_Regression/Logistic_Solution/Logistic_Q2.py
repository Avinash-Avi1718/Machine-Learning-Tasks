import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

#Importing Data
advertising = pd.read_csv("C:\\Users\Avinash\OneDrive\Desktop\DataScience_AI\Solutions\Logistic_Regression\\advertising.csv", sep = ",")

#Preprocessing
advertising.info()
advertising.Country.value_counts()
c1 = advertising.drop('Ad_Topic_Line', axis = 1)
c1 = c1.drop('Timestamp', axis = 1)
c1 = c1.drop('City', axis = 1)
lb = LabelEncoder()
c1['Country'] = lb.fit_transform(c1['Country'])
c1.head(11)
c1.describe()
c1.isna().sum()
c1.columns[0]
column_names = []
for i in range(len(c1.columns)):
    column_names.append(c1.columns[i].replace(' ', '_'))
c1.columns = column_names
c1.info()
#############################################################

# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit('Clicked_on_Ad~Daily_Time__Spent__on_Site+Age +Area_Income +Daily_Internet_Usage+Male+Country', data = c1).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(c1.iloc[ :, 0:6 ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(c1.Clicked_on_Ad, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
c1["pred"] = np.zeros(len(c1))
# taking threshold value and above the prob value will be treated as correct value 
c1.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(c1["pred"], c1["Clicked_on_Ad"])
classification

### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(c1, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('Clicked_on_Ad~Daily_Time__Spent__on_Site+Age +Area_Income +Daily_Internet_Usage+Male+Country', data = train_data).fit()

#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(len(test_data))

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['Clicked_on_Ad'])
confusion_matrix

accuracy_test = (155 + 135)/(len(test_data)) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["Clicked_on_Ad"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["Clicked_on_Ad"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test

# prediction on train data
train_pred = model.predict(train_data)

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(len(train_data))

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['Clicked_on_Ad'])
confusion_matrx

accuracy_train = (343 + 337)/(len(train_data))
print(accuracy_train)