# Load the data
data <- read.csv(file.choose(), header = T)
#View(data)

# Exploratory data analysis
summary(data)

#install.packages("Hmisc")
library(Hmisc)
describe(data)
#?describe

#install.packages("lattice")
library("lattice") # dotplot is part of lattice package

# Graphical exploration
dotplot(data$YearsExperience, main = "Dot Plot of YearsExperience Circumferences")
dotplot(data$Salary, main = "Dot Plot of salary Areas")

#?boxplot
boxplot(data$YearsExperience, col = "dodgerblue4")
boxplot(data$Salary, col = "red", horizontal = T)

hist(data$YearsExperience)
hist(data$Salary)

# Normal QQ plot
qqnorm(data$YearsExperience)
qqline(data$YearsExperience)

qqnorm(data$Salary)
qqline(data$Salary)

hist(data$YearsExperience, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(data$YearsExperience))             # add a density estimate with defaults
lines(density(data$YearsExperience, adjust = 2), lty = "dotted")   # add another "smoother" density

hist(data$Salary, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(data$Salary))             # add a density estimate with defaults
lines(density(data$Salary, adjust = 2), lty = "dotted")   # add another "smoother" density

# Bivariate analysis
# Scatter plot
plot(data$YearsExperience, data$Salary, main = "Scatter Plot", col = "Dodgerblue4", 
     col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "YearsExperience Ciscumference", 
     ylab = "salary area", pch = 20)  # plot(x,y)

#?plot

## alternate simple command
plot(data$YearsExperience, data$Salary)

attach(data)

# Correlation Coefficient
cor(YearsExperience, Salary)

# Covariance
cov(YearsExperience, Salary)

# Linear Regression model
reg <- lm(Salary ~ YearsExperience, data = data) # Y ~ X
#?lm
summary(reg)

confint(reg, level = 0.95)
#?confint

pred <- predict(reg, interval = "predict")
pred <- as.data.frame(pred)

#View(pred)
#?predict

# ggplot for adding Regression line for data
library(ggplot2)

ggplot(data = data, aes(YearsExperience, Salary) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ x)

# Alternate way
ggplot(data = data, aes(x = YearsExperience, y = Salary)) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = data, aes(x = YearsExperience, y = pred$fit))

# Evaluation the model for fitness 
cor(pred$fit, data$Salary)

reg$residuals
rmse <- sqrt(mean(reg$residuals^2))
rmse


# Transformation Techniques

# input = log(x); output = y

plot(log(YearsExperience), Salary)
cor(log(YearsExperience), Salary)

reg_log <- lm(Salary ~ log(YearsExperience), data = data)
summary(reg_log)

confint(reg_log,level = 0.95)
pred <- predict(reg_log, interval = "predict")

pred <- as.data.frame(pred)
cor(pred$fit, data$Salary)

rmse <- sqrt(mean(reg_log$residuals^2))
rmse

# Regression line for data
ggplot(data = data, aes(log(YearsExperience), Salary) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ log(x))

# Alternate way
ggplot(data = data, aes(x = log(YearsExperience), y = Salary)) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = data, aes(x = log(YearsExperience), y = pred$fit))



# Log transformation applied on 'y'
# input = x; output = log(y)

plot(YearsExperience, log(Salary))
cor(YearsExperience, log(Salary))

reg_log1 <- lm(log(Salary) ~ YearsExperience, data = data)
summary(reg_log1)

predlog <- predict(reg_log1, interval = "predict")
predlog <- as.data.frame(predlog)
reg_log1$residuals
sqrt(mean(reg_log1$residuals^2))

pred <- exp(predlog)  # Antilog = Exponential function
pred <- as.data.frame(pred)
cor(pred$fit, data$Salary)

res_log1 = Salary - pred$fit
rmse <- sqrt(mean(res_log1^2))
rmse

# Regression line for data
ggplot(data = data, aes(YearsExperience, log(Salary)) ) +
  geom_point() + stat_smooth(method = lm, formula = log(y) ~ x)

# Alternate way
ggplot(data = data, aes(x = YearsExperience, y = log(Salary))) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = data, aes(x = YearsExperience, y = predlog$fit))


# Non-linear models = Polynomial models
# input = x & x^2 (2-degree) and output = log(y)

reg2 <- lm(log(Salary) ~ YearsExperience + I(YearsExperience*YearsExperience), data = data)
summary(reg2)

predlog <- predict(reg2, interval = "predict")
pred <- exp(predlog)

pred <- as.data.frame(pred)
cor(pred$fit, data$Salary)

res2 = Salary - pred$fit
rmse <- sqrt(mean(res2^2))
rmse

# Regression line for data
ggplot(data = data, aes(YearsExperience, log(Salary)) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))

# Alternate way
ggplot(data = data, aes(x = YearsExperience + I(YearsExperience*YearsExperience), y = log(AT))) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = data, aes(x = YearsExperience + I(YearsExperience^2), y = predlog$fit))


# Data Partition

# Random Sampling
n <- nrow(data)
n1 <- n * 0.8
n2 <- n - n1

train_ind <- sample(1:n, n1)
train <- data[train_ind, ]
test <-  data[-train_ind, ]

# Non-random sampling
#train <- data[1:11, ]
#test <- data[11:14, ]

plot(train$YearsExperience, log(train$Salary))
plot(test$YearsExperience, log(test$Salary))

model <- lm(log(Salary) ~ YearsExperience + I(YearsExperience * YearsExperience), data = train)
summary(model)

confint(model,level=0.95)

log_res <- predict(model,interval = "confidence", newdata = test)

predict_original <- exp(log_res) # converting log values to original values
predict_original <- as.data.frame(predict_original)
test_error <- test$Salary - predict_original$fit # calculate error/residual
test_error

test_rmse <- sqrt(mean(test_error^2))
test_rmse

log_res_train <- predict(model, interval = "confidence", newdata = train)

predict_original_train <- exp(log_res_train) # converting log values to original values
predict_original_train <- as.data.frame(predict_original_train)
train_error <- train$Salary - predict_original_train$fit # calculate error/residual
train_error

train_rmse <- sqrt(mean(train_error^2))
train_rmse