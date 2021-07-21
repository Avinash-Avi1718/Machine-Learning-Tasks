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
dotplot(data$GPA, main = "Dot Plot of GPA Circumferences")
dotplot(data$SAT_Scores, main = "Dot Plot of SAT scores Areas")

#?boxplot
boxplot(data$GPA, col = "dodgerblue4")
boxplot(data$SAT_Scores, col = "red", horizontal = T)

hist(data$GPA)
hist(data$SAT_Scores)

# Normal QQ plot
qqnorm(data$GPA)
qqline(data$GPA)

qqnorm(data$SAT_Scores)
qqline(data$SAT_Scores)

hist(data$GPA, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(data$GPA))             # add a density estimate with defaults
lines(density(data$GPA, adjust = 2), lty = "dotted")   # add another "smoother" density

hist(data$SAT_Scores, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(data$SAT_Scores))             # add a density estimate with defaults
lines(density(data$SAT_Scores, adjust = 2), lty = "dotted")   # add another "smoother" density

# Bivariate analysis
# Scatter plot
plot(data$GPA, data$SAT_Scores, main = "Scatter Plot", col = "Dodgerblue4", 
     col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "GPA Ciscumference", 
     ylab = "SAT scores area", pch = 20)  # plot(x,y)

#?plot

## alternate simple command
plot(data$GPA, data$SAT_Scores)

attach(data)

# Correlation Coefficient
cor(GPA, SAT_Scores)

# Covariance
cov(GPA, SAT_Scores)

# Linear Regression model
reg <- lm(SAT_Scores ~ GPA, data = data) # Y ~ X
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

ggplot(data = data, aes(GPA, SAT_Scores) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ x)

# Alternate way
ggplot(data = data, aes(x = GPA, y = SAT_Scores)) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = data, aes(x = GPA, y = pred$fit))

# Evaluation the model for fitness 
cor(pred$fit, data$SAT_Scores)

reg$residuals
rmse <- sqrt(mean(reg$residuals^2))
rmse


# Transformation Techniques

# input = log(x); output = y

plot(log(GPA), SAT_Scores)
cor(log(GPA), SAT_Scores)

reg_log <- lm(SAT_Scores ~ log(GPA), data = data)
summary(reg_log)

confint(reg_log,level = 0.95)
pred <- predict(reg_log, interval = "predict")

pred <- as.data.frame(pred)
cor(pred$fit, data$SAT_Scores)

rmse <- sqrt(mean(reg_log$residuals^2))
rmse

# Regression line for data
ggplot(data = data, aes(log(GPA), SAT_Scores) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ log(x))

# Alternate way
ggplot(data = data, aes(x = log(GPA), y = SAT_Scores)) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = data, aes(x = log(GPA), y = pred$fit))


# Log transformation applied on 'y'
# input = x; output = log(y)

plot(GPA, log(SAT_Scores))
cor(GPA, log(SAT_Scores))

reg_log1 <- lm(log(SAT_Scores) ~ GPA, data = data)
summary(reg_log1)

predlog <- predict(reg_log1, interval = "predict")
predlog <- as.data.frame(predlog)
reg_log1$residuals
sqrt(mean(reg_log1$residuals^2))

pred <- exp(predlog)  # Antilog = Exponential function
pred <- as.data.frame(pred)
cor(pred$fit, data$SAT_Scores)

res_log1 = SAT_Scores - pred$fit
rmse <- sqrt(mean(res_log1^2))
rmse

# Regression line for data
ggplot(data = data, aes(GPA, log(SAT_Scores)) ) +
  geom_point() + stat_smooth(method = lm, formula = log(y) ~ x)

# Alternate way
ggplot(data = data, aes(x = GPA, y = log(SAT_Scores))) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = data, aes(x = GPA, y = predlog$fit))


# Non-linear models = Polynomial models
# input = x & x^2 (2-degree) and output = log(y)

reg2 <- lm(log(SAT_Scores) ~ GPA + I(GPA*GPA), data = data)
summary(reg2)

predlog <- predict(reg2, interval = "predict")
pred <- exp(predlog)

pred <- as.data.frame(pred)
cor(pred$fit, data$SAT_Scores)

res2 = SAT_Scores - pred$fit
rmse <- sqrt(mean(res2^2))
rmse

# Regression line for data
ggplot(data = data, aes(GPA, log(SAT_Scores)) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))

# Alternate way
ggplot(data = data, aes(x = GPA + I(GPA*GPA), y = log(AT))) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = data, aes(x = GPA + I(GPA^2), y = predlog$fit))


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

plot(train$GPA, log(train$SAT_Scores))
plot(test$GPA, log(test$SAT_Scores))

model <- lm(log(SAT_Scores) ~ GPA + I(GPA * GPA), data = train)
summary(model)

confint(model,level=0.95)

log_res <- predict(model,interval = "confidence", newdata = test)

predict_original <- exp(log_res) # converting log values to original values
predict_original <- as.data.frame(predict_original)
test_error <- test$SAT_Scores - predict_original$fit # calculate error/residual
test_error

test_rmse <- sqrt(mean(test_error^2))
test_rmse

log_res_train <- predict(model, interval = "confidence", newdata = train)

predict_original_train <- exp(log_res_train) # converting log values to original values
predict_original_train <- as.data.frame(predict_original_train)
train_error <- train$SAT_Scores - predict_original_train$fit # calculate error/residual
train_error

train_rmse <- sqrt(mean(train_error^2))
train_rmse