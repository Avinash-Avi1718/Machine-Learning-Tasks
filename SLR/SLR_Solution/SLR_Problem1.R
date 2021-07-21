# Load the data
data <- read.csv(file.choose(), header = T)
View(data)

# Exploratory data analysis
summary(data)

install.packages("Hmisc")
library(Hmisc)
describe(data)
#?describe

install.packages("lattice")
library("lattice") # dotplot is part of lattice package

# Graphical exploration
dotplot(data$Weight.gained..grams., main = "Dot Plot of Weight.gained..grams. Circumferences")
dotplot(data$Calories.Consumed, main = "Dot Plot of calories consumed Areas")

#?boxplot
boxplot(data$Weight.gained..grams., col = "dodgerblue4")
boxplot(data$Calories.Consumed, col = "red", horizontal = T)

hist(data$Weight.gained..grams.)
hist(data$Calories.Consumed)

# Normal QQ plot
qqnorm(data$Weight.gained..grams.)
qqline(data$Weight.gained..grams.)

qqnorm(data$Calories.Consumed)
qqline(data$Calories.Consumed)

hist(data$Weight.gained..grams., prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(data$Weight.gained..grams.))             # add a density estimate with defaults
lines(density(data$Weight.gained..grams., adjust = 2), lty = "dotted")   # add another "smoother" density

hist(data$Calories.Consumed, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(data$Calories.Consumed))             # add a density estimate with defaults
lines(density(data$Calories.Consumed, adjust = 2), lty = "dotted")   # add another "smoother" density

# Bivariate analysis
# Scatter plot
plot(data$Weight.gained..grams., data$Calories.Consumed, main = "Scatter Plot", col = "Dodgerblue4", 
     col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "Weight.gained..grams. Ciscumference", 
     ylab = "Calories consumed area", pch = 20)  # plot(x,y)

#?plot

## alternate simple command
plot(data$Weight.gained..grams., data$Calories.Consumed)

attach(data)

# Correlation Coefficient
cor(Weight.gained..grams., Calories.Consumed)

# Covariance
cov(Weight.gained..grams., Calories.Consumed)

# Linear Regression model
reg <- lm(Calories.Consumed ~ Weight.gained..grams., data = data) # Y ~ X
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

ggplot(data = data, aes(Weight.gained..grams., Calories.Consumed) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ x)

# Alternate way
ggplot(data = data, aes(x = Weight.gained..grams., y = Calories.Consumed)) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = data, aes(x = Weight.gained..grams., y = pred$fit))

# Evaluation the model for fitness 
cor(pred$fit, data$Calories.Consumed)

reg$residuals
rmse <- sqrt(mean(reg$residuals^2))
rmse


# Transformation Techniques

# input = log(x); output = y

plot(log(Weight.gained..grams.), Calories.Consumed)
cor(log(Weight.gained..grams.), Calories.Consumed)

reg_log <- lm(Calories.Consumed ~ log(Weight.gained..grams.), data = data)
summary(reg_log)

confint(reg_log,level = 0.95)
pred <- predict(reg_log, interval = "predict")

pred <- as.data.frame(pred)
cor(pred$fit, data$Calories.Consumed)

rmse <- sqrt(mean(reg_log$residuals^2))
rmse

# Regression line for data
ggplot(data = data, aes(log(Weight.gained..grams.), Calories.Consumed) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ log(x))

# Alternate way
ggplot(data = data, aes(x = log(Weight.gained..grams.), y = Calories.Consumed)) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = data, aes(x = log(Weight.gained..grams.), y = pred$fit))



# Log transformation applied on 'y'
# input = x; output = log(y)

plot(Weight.gained..grams., log(Calories.Consumed))
cor(Weight.gained..grams., log(Calories.Consumed))

reg_log1 <- lm(log(Calories.Consumed) ~ Weight.gained..grams., data = data)
summary(reg_log1)

predlog <- predict(reg_log1, interval = "predict")
predlog <- as.data.frame(predlog)
reg_log1$residuals
sqrt(mean(reg_log1$residuals^2))

pred <- exp(predlog)  # Antilog = Exponential function
pred <- as.data.frame(pred)
cor(pred$fit, data$Calories.Consumed)

res_log1 = Calories.Consumed - pred$fit
rmse <- sqrt(mean(res_log1^2))
rmse

# Regression line for data
ggplot(data = data, aes(Weight.gained..grams., log(Calories.Consumed)) ) +
  geom_point() + stat_smooth(method = lm, formula = log(y) ~ x)

# Alternate way
ggplot(data = data, aes(x = Weight.gained..grams., y = log(Calories.Consumed))) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = data, aes(x = Weight.gained..grams., y = predlog$fit))


# Non-linear models = Polynomial models
# input = x & x^2 (2-degree) and output = log(y)

reg2 <- lm(log(Calories.Consumed) ~ Weight.gained..grams. + I(Weight.gained..grams.*Weight.gained..grams.), data = data)
summary(reg2)

predlog <- predict(reg2, interval = "predict")
pred <- exp(predlog)

pred <- as.data.frame(pred)
cor(pred$fit, data$Calories.Consumed)

res2 = Calories.Consumed - pred$fit
rmse <- sqrt(mean(res2^2))
rmse

# Regression line for data
ggplot(data = data, aes(Weight.gained..grams., log(Calories.Consumed)) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))

# Alternate way
ggplot(data = data, aes(x = Weight.gained..grams. + I(Weight.gained..grams.*Weight.gained..grams.), y = log(AT))) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = data, aes(x = Weight.gained..grams. + I(Weight.gained..grams.^2), y = predlog$fit))


# Data Partition

# Random Sampling
n <- nrow(data)
n1 <- n * 0.8
n2 <- n - n1

train_ind <- sample(1:n, n1)
train <- data[train_ind, ]
test <-  data[-train_ind, ]

# Non-random sampling
train <- data[1:11, ]
test <- data[11:14, ]

plot(train$Weight.gained..grams., log(train$Calories.Consumed))
plot(test$Weight.gained..grams., log(test$Calories.Consumed))

model <- lm(log(Calories.Consumed) ~ Weight.gained..grams. + I(Weight.gained..grams. * Weight.gained..grams.), data = train)
summary(model)

confint(model,level=0.95)

log_res <- predict(model,interval = "confidence", newdata = test)

predict_original <- exp(log_res) # converting log values to original values
predict_original <- as.data.frame(predict_original)
test_error <- test$Calories.Consumed - predict_original$fit # calculate error/residual
test_error

test_rmse <- sqrt(mean(test_error^2))
test_rmse

log_res_train <- predict(model, interval = "confidence", newdata = train)

predict_original_train <- exp(log_res_train) # converting log values to original values
predict_original_train <- as.data.frame(predict_original_train)
train_error <- train$Calories.Consumed - predict_original_train$fit # calculate error/residual
train_error

train_rmse <- sqrt(mean(train_error^2))
train_rmse