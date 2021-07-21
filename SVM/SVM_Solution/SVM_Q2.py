# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:28:13 2021

@author: Avinash
"""
import pandas as pd
import numpy as np

data = pd.read_csv("D:\\Avinash\DataScience_AI\Solutions\SVM\\forestfires.csv")
data.info()
v_temp = data.dtypes
v_temp1 = v_temp['month']

data.columns

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

for i in data.columns:
    if v_temp[i] == v_temp1:
        data[i] = encoder.fit_transform(data[i])

data.info()

data.describe()
v_temp = data.head(5)
v_temp = data.size_category.value_counts()

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(data.iloc[:, 0:30],data.iloc[:, 30], test_size = 0.20)

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