# Load the Dataset
data <- read.csv(file.choose()) # Choose the data Data set

sum(is.na(data))

# Omitting NA values from the Data 
data1 <- na.omit(data) # na.omit => will omit the rows which has atleast 1 NA value
dim(data1)

# Alternatively We can apply mean/median/mode imputation
#data1 <- data # work with original data for imputation
summary(data1)

# NA values are present in CLMSEX, CLMINSUR, SEATBELT, CLMAGE

# Mean imputation for continuous data - CLMAGE
#data1$CLMAGE[is.na(data1$CLMAGE)] <- mean(data1$CLMAGE, na.rm = TRUE)


# Mode imputation for categorical data
# Custom function to calculate Mode
#Mode <- function(x){
#  a = table(x) # x is a vector
#  names(a[which.max(a)])
#}

#data1$CLMSEX[is.na(data1$CLMSEX)] <- Mode(data1$CLMSEX[!is.na(data1$CLMSEX)])
#data1$CLMINSUR[is.na(data1$CLMINSUR)] <- Mode(data1$CLMINSUR[!is.na(data1$CLMINSUR)])
#data1$SEATBELT[is.na(data1$SEATBELT)] <- Mode(data1$SEATBELT[!is.na(data1$SEATBELT)])

# We can also use imputeMissings package for imputation

sum(is.na(data1))
dim(data1)
###########

colnames(data)
data <- data1
#data <- data[ , c(-5,-6,-8,-9)] # Removing the first column which is is an Index

# Preparing a linear regression 
mod_lm <- lm(y ~ ., data = data)
summary(mod_lm)

pred1 <- predict(mod_lm, data)
pred1
# plot(data$CLMINSUR, pred1)

# We can also include NA values but where ever it finds NA value
# probability values obtained using the glm will also be NA 
# So they can be either filled using imputation technique or
# exlclude those values 


# GLM function use sigmoid curve to produce desirable ys 
# The output of sigmoid function lies in between 0-1
model <- glm(y ~ ., data = data, family = "binomial")
summary(model)
# We are going to use NULL and Residual Deviance to compare the between different models

# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))

# Prediction to check model validation
prob <- predict(model, data, type = "response")
prob

# or use plogis for prediction of probabilities
prob <- plogis(predict(model, data))

# Confusion matrix and considering the threshold value as 0.5 
confusion <- table(prob > 0.5, data$y)
confusion

# Model Accuracy 
Acc <- sum(diag(confusion)/sum(confusion))
Acc

# Convert the probabilities to binary output form using cutoff
pred_values <- ifelse(prob > 0.5, 1, 0)

library(caret)
# Confusion Matrix
confusionMatrix(factor(data$y, levels = c(0, 1)), factor(pred_values, levels = c(0, 1)))


# Build Model on 100% of data
data1 <- data # Removing the first column which is is an Index
# To Find the optimal Cutoff value:
# The output of sigmoid function lies in between 0-1
fullmodel <- glm(y ~ ., data = data1, family = "binomial")
summary(fullmodel)

prob_full <- predict(fullmodel, data1, type = "response")
prob_full

# Decide on optimal prediction probability cutoff for the model
library(InformationValue)
optCutOff <- optimalCutoff(data1$y, prob_full)
optCutOff

# Check multicollinearity in the model
library(car)
vif(fullmodel)

# Misclassification Error - the percentage mismatch of predcited vs actuals
# Lower the misclassification error, better the model.
misClassError(data1$y, prob_full, threshold = optCutOff)


# ROC curve
# Greater the area under the ROC curve, better the predictive ability of the model
plotROC(data1$y, prob_full)


# Confusion Matrix
predvalues <- ifelse(prob_full > optCutOff, 1, 0)

ys <- confusionMatrix(predvalues, data1$y)

sensitivity(predvalues, data1$y)
confusionMatrix(actuals = data1$y, predictedScores = predvalues)


###################
# Data Partitioning
n <- nrow(data1)
n1 <- n * 0.85
n2 <- n - n1
train_index <- sample(1:n, n1)
train <- data1[train_index, ]
test <- data1[-train_index, ]

# Train the model using Training data
finalmodel <- glm(y ~ ., data = train, family = "binomial")
summary(finalmodel)

# Prediction on test data
prob_test <- predict(finalmodel, newdata = test, type = "response")
prob_test

# Confusion matrix 
confusion <- table(prob_test > optCutOff, test$y)
confusion

# Model Accuracy 
Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy 

# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
pred_values <- ifelse(prob_test > optCutOff, 1, 0)

# Creating new column to store the above values
test[,"prob"] <- prob_test
test[,"pred_values"] <- pred_values

table(test$y, test$pred_values)


# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(finalmodel, newdata = train, type = "response")
prob_train

# Confusion matrix
confusion_train <- table(prob_train > optCutOff, train$y)
confusion_train

# Model Accuracy 
Acc_train <- sum(diag(confusion_train)/sum(confusion_train))
Acc_train