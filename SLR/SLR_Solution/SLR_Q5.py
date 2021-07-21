# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
from sklearn import preprocessing

df = pd.read_csv("C:\\Users\\Avinash\\OneDrive\\Desktop\\DataScience_AI\\Solutions\\SLR\\SAT_GPA.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.replace('(', '')
df.columns = df.columns.str.replace(')', '')
df.describe()
df.head()
df.info()
#Graphical Representation
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

min(df.GPA)
max(df.GPA)

plt.bar(height = df.GPA, x = np.arange(2, 4, 0.2))
plt.hist(df.GPA) #histogram
plt.boxplot(df.GPA) #boxplot

plt.bar(height = df.SAT_Scores, x = np.arange(1, 11, 1))
plt.hist(df.SAT_Scores) #histogram
plt.boxplot(df.SAT_Scores) #boxplot

# Scatter plot
plt.scatter(x = df['SAT_Scores'], y = df['GPA'], color = 'green') 
'''
#Applying normalization
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
'''
np.corrcoef(df.SAT_Scores, df.GPA) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance mGPArix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(df.SAT_Scores, df.GPA)[0, 1]
cov_output

# df.cov()

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('SAT_Scores ~ GPA', data = df).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(df['GPA']))
pred1

# Regression Line
plt.scatter(df.SAT_Scores, df.GPA)
plt.plot(df.GPA, pred1, "r")
plt.legend(['Predicted line', 'Observed dGPAa'])
plt.show()

# Error calculation
res1 = df.SAT_Scores - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(SAT_Scores); y = GPA

plt.scatter(x = np.log(df['GPA']), y = df['SAT_Scores'], color = 'brown')
np.corrcoef(np.log(df.GPA), df.SAT_Scores) #correlation

model2 = smf.ols('SAT_Scores ~ np.log(GPA)', data = df).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(df['GPA']))
pred2

# Regression Line
plt.scatter(np.log(df.GPA), df.SAT_Scores)
plt.plot(np.log(df.GPA), pred2, "r")
plt.legend(['Predicted line', 'Observed dGPAa'])
plt.show()

# Error calculGPAion
res2 = df.GPA - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Polynomial transformGPAion
# x = SAT_Scores; x^2 = SAT_Scores*SAT_Scores; y = log(GPA)

model4 = smf.ols('np.log(SAT_Scores) ~ GPA + I(GPA*GPA)', data = df).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(df['GPA']))
pred4_GPA = np.exp(pred4)
pred4_GPA

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = df.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = df.iloc[:, 1].values

plt.scatter(df.SAT_Scores, np.log(df.GPA))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed dGPAa'])
plt.show()

# Error calculation
res4 = df.GPA - pred4_GPA
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size = 0.2)

finalmodel = smf.ols('np.log(SAT_Scores) ~ GPA + I(GPA*GPA)', data = train).fit()
finalmodel.summary()

# Predict on test dGPAa
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_GPA = np.exp(test_pred)
pred_test_GPA

# Model EvaluGPAion on Test dGPAa
test_res = test.GPA - pred_test_GPA
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

# Prediction on train dGPAa
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_GPA = np.exp(train_pred)
pred_train_GPA

# Model EvaluGPAion on train dGPAa
train_res = train.GPA - pred_train_GPA
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse