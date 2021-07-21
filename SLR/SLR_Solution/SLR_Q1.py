# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

df = pd.read_csv("C:\\Users\\Avinash\\OneDrive\\Desktop\\DataScience_AI\\Solutions\\SLR\\calories_consumed.csv")

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

plt.bar(height = df.Calories_Consumed, x = np.arange(1, 15, 1))
plt.hist(df.Calories_Consumed) #histogram
plt.boxplot(df.Calories_Consumed) #boxplot

plt.bar(height = df.Weight_gained_grams, x = np.arange(1, 15, 1))
plt.hist(df.Weight_gained_grams) #histogram
plt.boxplot(df.Weight_gained_grams) #boxplot

# Scatter plot
plt.scatter(x = df['Weight_gained_grams'], y = df['Calories_Consumed'], color = 'green') 

plt.scatter(np.log(df.Weight_gained_grams), df.Calories_Consumed)

# correlation
np.corrcoef(df.Weight_gained_grams, df.Calories_Consumed) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance mCalories_Consumedrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(df.Weight_gained_grams, df.Calories_Consumed)[0, 1]
cov_output

# df.cov()


# Import library
import statsmodels.formula.api as smf
help(smf.ols)
# Simple Linear Regression
model = smf.ols('Calories_Consumed ~ Weight_gained_grams', data = df).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(df['Weight_gained_grams']))
pred1

# Regression Line
plt.scatter(df.Weight_gained_grams, df.Calories_Consumed)
plt.plot(df.Weight_gained_grams, pred1, "r")
plt.legend(['Predicted line', 'Observed Calories_Consumed'])
plt.show()

# Error calculation
res1 = df.Calories_Consumed - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Calories_Consumed
# Log Transformation
# x = log(Weight_gained_grams); y = Calories_Consumed

plt.scatter(x = np.log(df['Weight_gained_grams']), y = df['Calories_Consumed'], color = 'brown')
np.corrcoef(np.log(df.Weight_gained_grams), df.Calories_Consumed) #correlation

model2 = smf.ols('Calories_Consumed ~ np.log(Weight_gained_grams)', data = df).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(df['Weight_gained_grams']))
pred2

# Regression Line
plt.scatter(np.log(df.Weight_gained_grams), df.Calories_Consumed)
plt.plot(np.log(df.Weight_gained_grams), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = df.Calories_Consumed - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformCalories_Consumption
# x = Weight_gained_grams; y = log(Calories_Consumed)

plt.scatter(x = df['Weight_gained_grams'], y = np.log(df['Calories_Consumed']), color = 'orange')
np.corrcoef(df.Weight_gained_grams, np.log(df.Calories_Consumed)) #correlCalories_Consumedion

model3 = smf.ols('np.log(Calories_Consumed) ~ Weight_gained_grams', data = df).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(df['Weight_gained_grams']))
pred3_Calories_Consumed = np.exp(pred3)
pred3_Calories_Consumed

# Regression Line
plt.scatter(df.Weight_gained_grams, np.log(df.Calories_Consumed))
plt.plot(df.Weight_gained_grams, pred3, "r")
plt.legend(['Predicted line', 'Observed Calories_Consumed'])
plt.show()

# Error calculCalories_Consumedion
res3 = df.Calories_Consumed - pred3_Calories_Consumed
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = Weight_gained_grams; x^2 = Weight_gained_grams*Weight_gained_grams; y = log(Calories_Consumed)

model4 = smf.ols('np.log(Calories_Consumed) ~ Weight_gained_grams + I(Weight_gained_grams*Weight_gained_grams)', data = df).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(df))
pred4_Calories_Consumed = np.exp(pred4)
pred4_Calories_Consumed

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = df.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = df.iloc[:, 1].values


plt.scatter(df.Weight_gained_grams, np.log(df.Calories_Consumed))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = df.Calories_Consumed - pred4_Calories_Consumed
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

finalmodel = smf.ols('Calories_Consumed ~ Weight_gained_grams', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_Calories_Consumed = np.exp(test_pred)
pred_test_Calories_Consumed

# Model Evaluation on Test data
test_res = test.Calories_Consumed - pred_test_Calories_Consumed
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_Calories_Consumed = np.exp(train_pred)
pred_train_Calories_Consumed

# Model Evaluation on train data
train_res = train.Calories_Consumed - pred_train_Calories_Consumed
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse