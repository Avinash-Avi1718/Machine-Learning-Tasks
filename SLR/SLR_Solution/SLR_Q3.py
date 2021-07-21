# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

df = pd.read_csv("C:\\Users\\Avinash\\OneDrive\\Desktop\\DataScience_AI\\Solutions\\SLR\\emp_data.csv")

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

plt.bar(height = df.Churn_out_rate, x = np.arange(1, 11, 1))
plt.hist(df.Churn_out_rate) #histogram
plt.boxplot(df.Churn_out_rate) #boxplot

plt.bar(height = df.Salary_hike, x = np.arange(1, 11, 1))
plt.hist(df.Salary_hike) #histogram
plt.boxplot(df.Salary_hike) #boxplot

# Scatter plot
plt.scatter(x = df['Salary_hike'], y = df['Churn_out_rate'], color = 'green') 

# correlation
np.corrcoef(df.Salary_hike, df.Churn_out_rate) 

# Covariance
# NumPy does not have a function to calculChurn_out_ratee the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(df.Salary_hike, df.Churn_out_rate)[0, 1]
cov_output

# df.cov()

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('Churn_out_rate ~ Salary_hike', data = df).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(df['Salary_hike']))
pred1

# Regression Line
plt.scatter(df.Salary_hike, df.Churn_out_rate)
plt.plot(df.Salary_hike, pred1, "r")
plt.legend(['Predicted line', 'Observed dChurn_out_ratea'])
plt.show()

# Error calculation
res1 = df.Churn_out_rate - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(Salary_hike); y = Churn_out_rate

plt.scatter(x = np.log(df['Salary_hike']), y = df['Churn_out_rate'], color = 'brown')
np.corrcoef(np.log(df.Salary_hike), df.Churn_out_rate) #correlation

model2 = smf.ols('Churn_out_rate ~ np.log(Salary_hike)', data = df).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(df['Salary_hike']))
pred2

# Regression Line
plt.scatter(np.log(df.Salary_hike), df.Churn_out_rate)
plt.plot(np.log(df.Salary_hike), pred2, "r")
plt.legend(['Predicted line', 'Observed dChurn_out_ratea'])
plt.show()

# Error calculation
res2 = df.Churn_out_rate - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = Salary_hike; y = log(Churn_out_rate)

plt.scatter(x = df['Salary_hike'], y = np.log(df['Churn_out_rate']), color = 'orange')
np.corrcoef(df.Salary_hike, np.log(df.Churn_out_rate)) #correlChurn_out_rateion

model3 = smf.ols('np.log(Churn_out_rate) ~ Salary_hike', data = df).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(df['Salary_hike']))
pred3_Churn_out_rate = np.exp(pred3)
pred3_Churn_out_rate

# Regression Line
plt.scatter(df.Salary_hike, np.log(df.Churn_out_rate))
plt.plot(df.Salary_hike, pred3, "r")
plt.legend(['Predicted line', 'Observed dChurn_out_ratea'])
plt.show()

# Error calculation
res3 = df.Churn_out_rate - pred3_Churn_out_rate
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = Salary_hike; x^2 = Salary_hike*Salary_hike; y = log(Churn_out_rate)

model4 = smf.ols('np.log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike)', data = df).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(df))
pred4_Churn_out_rate = np.exp(pred4)
pred4_Churn_out_rate

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = df.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = df.iloc[:, 1].values


plt.scatter(df.Salary_hike, np.log(df.Churn_out_rate))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed dChurn_out_ratea'])
plt.show()


# Error calculation
res4 = df.Churn_out_rate - pred4_Churn_out_rate
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size = 0.2)

finalmodel = smf.ols('np.log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_Churn_out_rate = np.exp(test_pred)
pred_test_Churn_out_rate

# Model Evaluation on Test data
test_res = test.Churn_out_rate - pred_test_Churn_out_rate
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_Churn_out_rate = np.exp(train_pred)
pred_train_Churn_out_rate

# Model Evaluation on train data
train_res = train.Churn_out_rate - pred_train_Churn_out_rate
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse