# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:22:51 2021

@author: Avinash
"""
import pandas as pd
import numpy as np

#letters = pd.read_csv("C:/Users/Vybhav Uttarkar/Desktop/DS_and_AI/28. Black Box Technique - SVM/letterdata.csv/letterdata.csv")
train = pd.read_csv("D:\\Avinash\DataScience_AI\Solutions\SVM\\SalaryData_Train (1).csv")
test = pd.read_csv("D:\\Avinash\DataScience_AI\Solutions\SVM\\SalaryData_Test (1).csv")
train = train.iloc[0:500,0:]
test = test.iloc[0:500, 0:]

v_temp = train.dtypes
v_temp1 = v_temp['sex']

train.columns

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train.workclass = encoder.fit_transform(train.workclass)

for i in train.columns:
    if v_temp[i] == v_temp1:
        train[i] = encoder.fit_transform(train[i])
        test[i] = encoder.fit_transform(test[i])

test.info()

train.describe()
v_temp = test.head(5)
v_temp = train.Salary.value_counts()

from sklearn.svm import SVC
#from sklearn.model_selection import train_test_split

#train_X, test_X, train_y, test_y = train_test_split(letters.iloc[:, 1:],letters.iloc[:, 0], test_size = 0.20)

train_X = train.iloc[:, 0:13]
train_y = train.iloc[:, 13:]
test_X  = test.iloc[:, 0:13]
test_y  = test.iloc[:, 13:]

# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X, train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear == np.array(test_y))

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==np.array(test_y))

