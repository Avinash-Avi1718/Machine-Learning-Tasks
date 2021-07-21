library(readr)

#Importing the dataset
glass <- read_csv("C:\\Users\\Avinash\\Desktop\\DataScience_AI\\Solutions\\KNN\\glass.csv")
#glass <- glass[, c(10,1,2,3,4,5,6,7,8,9)]

# Exploratory Data Analysis
# table of Type
table(glass$Type)

str(glass$Type)
# recode Type as a factor
glass$Type <- factor(glass$Type, levels = c("1", "2", "3", "5", "6", "7"))

# table or proportions with more informative labels
round(prop.table(table(glass$Type)) * 100, digits = 2)

# summarize any three numeric features
summary(glass[c("RI", "Na", "Mg")])

# create normalization function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# test normalization function - result should be identical
normalize(c(0.01, 0.02, 0.03, 0.04, 0.05))
normalize(c(10, 20, 30, 40, 50))

# normalize the glass data
glass_n <- as.data.frame(lapply(glass[1:9], normalize))

# confirm that normalization worked
summary(glass_n$RI)

# create training and test data
glass_train <- glass_n[1:150, ]
glass_test <- glass_n[151:214, ]

# create labels for training and test data
glass_train_labels <- glass[1:150, 10]
glass_train_labels <- glass_train_labels[["Type"]] 

glass_test_labels <- glass[151:214, 10]
glass_test_labels <- glass_test_labels[["Type"]]
#---- Training a model on the data ----

# load the "class" library
install.packages("class")
library(class)
glass_test_pred <- knn(train = glass_train, test = glass_test,
                      cl = glass_train_labels, k = 14)


## ---- Evaluating model performance ---- ##
confusion_test <- table(x = glass_test_labels, y = glass_test_pred)
confusion_test

Accuracy <- sum(diag(confusion_test))/sum(confusion_test)
Accuracy 

# Training Accuracy to compare against test accuracy
glass_train_pred <- knn(train = glass_train, test = glass_train, cl = glass_train_labels, k=14)

confusion_train <- table(x = glass_train_labels, y = glass_train_pred)
confusion_train

Accuracy_train <- sum(diag(confusion_train))/sum(confusion_train)
Accuracy_train

## Improving model performance ----

# use the scale() function to z-score standardize a data frame
glass_z <- as.data.frame(scale(glass[-10]))

# confirm that the transformation was applied correctly
summary(glass_z$RI)

# create training and test datasets
glass_train <- glass_z[1:150, ]
glass_test <- glass_z[151:214, ]

# re-classify test cases
glass_test_pred <- knn(train = glass_train, test = glass_test,
                      cl = glass_train_labels, k=10)

# Create the cross tabulation of predicted vs. actual
library(gmodels)
CrossTable(x = glass_test_labels, y = glass_test_pred, prop.chisq=FALSE)


# try several different values of k
glass_train <- glass_n[1:150, ]
glass_test <- glass_n[151:214, ]

glass_test_pred <- knn(train = glass_train, test = glass_test, cl = glass_train_labels, k=1)
CrossTable(x = glass_test_labels, y = glass_test_pred, prop.chisq=FALSE)

glass_test_pred <- knn(train = glass_train, test = glass_test, cl = glass_train_labels, k=5)
CrossTable(x = glass_test_labels, y = glass_test_pred, prop.chisq=FALSE)

glass_test_pred <- knn(train = glass_train, test = glass_test, cl = glass_train_labels, k=11)
CrossTable(x = glass_test_labels, y = glass_test_pred, prop.chisq=FALSE)

glass_test_pred <- knn(train = glass_train, test = glass_test, cl = glass_train_labels, k=15)
CrossTable(x = glass_test_labels, y = glass_test_pred, prop.chisq=FALSE)

glass_test_pred <- knn(train = glass_train, test = glass_test, cl = glass_train_labels, k=21)
CrossTable(x = glass_test_labels, y = glass_test_pred, prop.chisq=FALSE)

glass_test_pred <- knn(train = glass_train, test = glass_test, cl = glass_train_labels, k=27)
CrossTable(x = glass_test_labels, y = glass_test_pred, prop.chisq=FALSE)


########################################################
pred.train <- NULL
pred.val <- NULL
error_rate.train <- NULL
error_rate.val <- NULL
accu_rate.train <- NULL
accu_rate.val <- NULL
accu.diff <- NULL
error.diff <- NULL

for (i in 1:21) {
  pred.train <- knn(train = glass_train, test = glass_train, cl = glass_train_labels, k = i)
  pred.val <- knn(train = glass_train, test = glass_test, cl = glass_train_labels, k = i)
  error_rate.train[i] <- mean(pred.train!=glass_train_labels)
  error_rate.val[i] <- mean(pred.val != glass_test_labels)
  accu_rate.train[i] <- mean(pred.train == glass_train_labels)
  accu_rate.val[i] <- mean(pred.val == glass_test_labels)  
  accu.diff[i] = accu_rate.train[i] - accu_rate.val[i]
  error.diff[i] = error_rate.val[i] - error_rate.train[i]
}

knn.error <- as.data.frame(cbind(k = 1:21, error.train = error_rate.train, error.val = error_rate.val, error.diff = error.diff))
knn.accu <- as.data.frame(cbind(k = 1:21, accu.train = accu_rate.train, accu.val = accu_rate.val, accu.diff = accu.diff))

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
#From the graph 14 is the suitable k value

# Plot for Error
plot(knn.error[, c(4)], type = "b", xlab = "K-Value", ylab = "DifferenceInError")