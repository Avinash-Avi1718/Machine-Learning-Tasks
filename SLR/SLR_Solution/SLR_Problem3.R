# Load the data
data <- read.csv(file.choose(), header = T)
#View(data)

# Exploratory data analysis
summary(data)

install.packages("Hmisc")
library(Hmisc)
describe(data)
#?describe

install.packages("lattice")
library("lattice") # dotplot is part of lattice package

# Graphical exploration
dotplot(data$Salary_hike, main = "Dot Plot of Salary_hike Circumferences")
dotplot(data$Churn_out_rate, main = "Dot Plot of churn rate Areas")

#?boxplot
boxplot(data$Salary_hike, col = "dodgerblue4")
boxplot(data$Churn_out_rate, col = "red", horizontal = T)

hist(data$Salary_hike)
hist(data$Churn_out_rate)

# Normal QQ plot
qqnorm(data$Salary_hike)
qqline(data$Salary_hike)

qqnorm(data$Churn_out_rate)
qqline(data$Churn_out_rate)

hist(data$Salary_hike, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(data$Salary_hike))             # add a density estimate with defaults
lines(density(data$Salary_hike, adjust = 2), lty = "dotted")   # add another "smoother" density

hist(data$Churn_out_rate, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(data$Churn_out_rate))             # add a density estimate with defaults
lines(density(data$Churn_out_rate, adjust = 2), lty = "dotted")   # add another "smoother" density

# Bivariate analysis
# Scatter plot
plot(data$Salary_hike, data$Churn_out_rate, main = "Scatter Plot", col = "Dodgerblue4", 
     col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "Salary_hike Ciscumference", 
     ylab = "Churn rate area", pch = 20)  # plot(x,y)

#?plot

## alternate simple command
plot(data$Salary_hike, data$Churn_out_rate)

attach(data)

# Correlation Coefficient
cor(Salary_hike, Churn_out_rate)

# Covariance
cov(Salary_hike, Churn_out_rate)

# Linear Regression model
reg <- lm(Churn_out_rate ~ Salary_hike, data = data) # Y ~ X
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

ggplot(data = data, aes(Salary_hike, Churn_out_rate) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ x)

# Alternate way
ggplot(data = data, aes(x = Salary_hike, y = Churn_out_rate)) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = data, aes(x = Salary_hike, y = pred$fit))

# Evaluation the model for fitness 
cor(pred$fit, data$Churn_out_rate)

reg$residuals
rmse <- sqrt(mean(reg$residuals^2))
rmse


# Transformation Techniques

# input = log(x); output = y

plot(log(Salary_hike), Churn_out_rate)
cor(log(Salary_hike), Churn_out_rate)

reg_log <- lm(Churn_out_rate ~ log(Salary_hike), data = data)
summary(reg_log)

confint(reg_log,level = 0.95)
pred <- predict(reg_log, interval = "predict")

pred <- as.data.frame(pred)
cor(pred$fit, data$Churn_out_rate)

rmse <- sqrt(mean(reg_log$residuals^2))
rmse

# Regression line for data
ggplot(data = data, aes(log(Salary_hike), Churn_out_rate) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ log(x))

# Alternate way
ggplot(data = data, aes(x = log(Salary_hike), y = Churn_out_rate)) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = data, aes(x = log(Salary_hike), y = pred$fit))



# Log transformation applied on 'y'
# input = x; output = log(y)

plot(Salary_hike, log(Churn_out_rate))
cor(Salary_hike, log(Churn_out_rate))

reg_log1 <- lm(log(Churn_out_rate) ~ Salary_hike, data = data)
summary(reg_log1)

predlog <- predict(reg_log1, interval = "predict")
predlog <- as.data.frame(predlog)
reg_log1$residuals
sqrt(mean(reg_log1$residuals^2))

pred <- exp(predlog)  # Antilog = Exponential function
pred <- as.data.frame(pred)
cor(pred$fit, data$Churn_out_rate)

res_log1 = Churn_out_rate - pred$fit
rmse <- sqrt(mean(res_log1^2))
rmse

# Regression line for data
ggplot(data = data, aes(Salary_hike, log(Churn_out_rate)) ) +
  geom_point() + stat_smooth(method = lm, formula = log(y) ~ x)

# Alternate way
ggplot(data = data, aes(x = Salary_hike, y = log(Churn_out_rate))) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = data, aes(x = Salary_hike, y = predlog$fit))


# Non-linear models = Polynomial models
# input = x & x^2 (2-degree) and output = log(y)

reg2 <- lm(log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike), data = data)
summary(reg2)

predlog <- predict(reg2, interval = "predict")
pred <- exp(predlog)

pred <- as.data.frame(pred)
cor(pred$fit, data$Churn_out_rate)

res2 = Churn_out_rate - pred$fit
rmse <- sqrt(mean(res2^2))
rmse

# Regression line for data
ggplot(data = data, aes(Salary_hike, log(Churn_out_rate)) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))

# Alternate way
ggplot(data = data, aes(x = Salary_hike + I(Salary_hike*Salary_hike), y = log(AT))) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = data, aes(x = Salary_hike + I(Salary_hike^2), y = predlog$fit))


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

plot(train$Salary_hike, log(train$Churn_out_rate))
plot(test$Salary_hike, log(test$Churn_out_rate))

model <- lm(log(Churn_out_rate) ~ Salary_hike + I(Salary_hike * Salary_hike), data = train)
summary(model)

confint(model,level=0.95)

log_res <- predict(model,interval = "confidence", newdata = test)

predict_original <- exp(log_res) # converting log values to original values
predict_original <- as.data.frame(predict_original)
test_error <- test$Churn_out_rate - predict_original$fit # calculate error/residual
test_error

test_rmse <- sqrt(mean(test_error^2))
test_rmse

log_res_train <- predict(model, interval = "confidence", newdata = train)

predict_original_train <- exp(log_res_train) # converting log values to original values
predict_original_train <- as.data.frame(predict_original_train)
train_error <- train$Churn_out_rate - predict_original_train$fit # calculate error/residual
train_error

train_rmse <- sqrt(mean(train_error^2))
train_rmse