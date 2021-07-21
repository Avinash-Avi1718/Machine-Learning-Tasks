
# drop the animal name feature
Zoo <- Zoo[2:18]

# Exploratory Data Analysis

# table of type
table(Zoo$type)

# create normalization function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# normalize the Zoo data
Zoo_n <- as.data.frame(lapply(Zoo[1:16], normalize))

# confirm that normalization worked
summary(Zoo_n$hair)

# create training and test data
Zoo_train <- Zoo_n[1:70, ]
Zoo_test <- Zoo_n[71:101, ]

# create labels for training and test data

Zoo_train_labels <- Zoo[1:70, 17]
Zoo_train_labels <- Zoo_train_labels[["type"]] 

Zoo_test_labels <- Zoo[71:101, 17]
Zoo_test_labels <- Zoo_test_labels[["type"]]
#---- Training a model on the data ----

# load the "class" library
#install.packages("class")
library(class)

Zoo_test_pred <- knn(train = Zoo_train, test = Zoo_test,
                     cl = Zoo_train_labels, k = 3)


## ---- Evaluating model performance ---- ##
confusion_test <- table(x = Zoo_test_labels, y = Zoo_test_pred)
confusion_test

Accuracy <- sum(diag(confusion_test))/sum(confusion_test)
Accuracy 

# Training Accuracy to compare against test accuracy
Zoo_train_pred <- knn(train = Zoo_train, test = Zoo_train, cl = Zoo_train_labels, k=3)

confusion_train <- table(x = Zoo_train_labels, y = Zoo_train_pred)
confusion_train

Accuracy_train <- sum(diag(confusion_train))/sum(confusion_train)
Accuracy_train

## Improving model performance ----

# use the scale() function to z-score standardize a data frame
Zoo_z <- as.data.frame(scale(Zoo[1:16]))

# confirm that the transformation was applied correctly
summary(Zoo_z$hair)

# create training and test datasets
Zoo_train <- Zoo_z[1:70, ]
Zoo_test <- Zoo_z[71:101, ]

# re-classify test cases
Zoo_test_pred <- knn(train = Zoo_train, test = Zoo_test,
                     cl = Zoo_train_labels, k=3)

# Create the cross tabulation of predicted vs. actual
library(gmodels)
CrossTable(x = Zoo_test_labels, y = Zoo_test_pred, prop.chisq=FALSE)


# try several different values of k
Zoo_train <- Zoo_n[1:70, ]
Zoo_test <- Zoo_n[71:101, ]

Zoo_test_pred <- knn(train = Zoo_train, test = Zoo_test, cl = Zoo_train_labels, k=1)
CrossTable(x = Zoo_test_labels, y = Zoo_test_pred, prop.chisq=FALSE)

Zoo_test_pred <- knn(train = Zoo_train, test = Zoo_test, cl = Zoo_train_labels, k=5)
CrossTable(x = Zoo_test_labels, y = Zoo_test_pred, prop.chisq=FALSE)

Zoo_test_pred <- knn(train = Zoo_train, test = Zoo_test, cl = Zoo_train_labels, k=11)
CrossTable(x = Zoo_test_labels, y = Zoo_test_pred, prop.chisq=FALSE)

Zoo_test_pred <- knn(train = Zoo_train, test = Zoo_test, cl = Zoo_train_labels, k=15)
CrossTable(x = Zoo_test_labels, y = Zoo_test_pred, prop.chisq=FALSE)

Zoo_test_pred <- knn(train = Zoo_train, test = Zoo_test, cl = Zoo_train_labels, k=21)
CrossTable(x = Zoo_test_labels, y = Zoo_test_pred, prop.chisq=FALSE)

Zoo_test_pred <- knn(train = Zoo_train, test = Zoo_test, cl = Zoo_train_labels, k=27)
CrossTable(x = Zoo_test_labels, y = Zoo_test_pred, prop.chisq=FALSE)


########################################################
pred.train <- NULL
pred.val <- NULL
error_rate.train <- NULL
error_rate.val <- NULL
accu_rate.train <- NULL
accu_rate.val <- NULL
accu.diff <- NULL
error.diff <- NULL

for (i in 1:39) {
  pred.train <- knn(train = Zoo_train, test = Zoo_train, cl = Zoo_train_labels, k = i)
  pred.val <- knn(train = Zoo_train, test = Zoo_test, cl = Zoo_train_labels, k = i)
  error_rate.train[i] <- mean(pred.train!=Zoo_train_labels)
  error_rate.val[i] <- mean(pred.val != Zoo_test_labels)
  accu_rate.train[i] <- mean(pred.train == Zoo_train_labels)
  accu_rate.val[i] <- mean(pred.val == Zoo_test_labels)  
  accu.diff[i] = accu_rate.train[i] - accu_rate.val[i]
  error.diff[i] = error_rate.val[i] - error_rate.train[i]
}

knn.error <- as.data.frame(cbind(k = 1:39, error.train = error_rate.train, error.val = error_rate.val, error.diff = error.diff))
knn.accu <- as.data.frame(cbind(k = 1:39, accu.train = accu_rate.train, accu.val = accu_rate.val, accu.diff = accu.diff))

library(ggplot2)
errorPlot = ggplot() + 
  geom_line(data = knn.error[, -c(3,4)], aes(x = k, y = error.train), color = "blue") +
  geom_line(data = knn.error[, -c(2,4)], aes(x = k, y = error.val), color = "red") +
  geom_line(data = knn.error[, -c(2,3)], aes(x = k, y = error.diff), color = "black") +
  xlab('knn') +
  ylab('ErrorRate')
accuPlot = ggplot() + 
  geom_line(data = knn.accu[,-c(3,4)], aes(x = k, y = accu.train), color = "blue") +
  geom_line(data = knn.accu[,-c(2,4)], aes(x = k, y = accu.val), color = "red") +
  geom_line(data = knn.accu[,-c(2,3)], aes(x = k, y = accu.diff), color = "black") +
  xlab('knn') +
  ylab('AccuracyRate')

# Plot for Accuracy
plot(knn.accu[, c(4)], type = "b", xlab = "K-Value", ylab = "DifferenceInAccu") 
#From the curve k value is 16 is best suitable

# Plot for Error
plot(knn.error[, c(4)], type = "b", xlab = "K-Value", ylab = "DifferenceInError") 