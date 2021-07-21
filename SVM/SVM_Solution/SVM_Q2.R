#####Support Vector Machines 

# Load the Dataset
data <- read.csv(file.choose(), stringsAsFactors = TRUE)

summary(data)

# Partition Data into train and test data
data_train <- data[1:400, ]
data_test  <- data[401:517, ]

# Training a model on the data ----
# Begin by training a simple linear SVM
#install.packages("kernlab")
library(kernlab)

data_classifier <- ksvm(size_category ~ ., data = data_train, kernel = "vanilladot")
#?ksvm

## Evaluating model performance ----
# predictions on testing dataset
data_predictions <- predict(data_classifier, data_test)

table(data_predictions, data_test$size_category)
agreement <- data_predictions == data_test$size_category
table(agreement)
prop.table(table(agreement))

## Improving model performance ----
data_classifier_rbf <- ksvm(size_category ~ ., data = data_train, kernel = "rbfdot")
data_predictions_rbf <- predict(data_classifier_rbf, data_test)
agreement_rbf <- data_predictions_rbf == data_test$size_category
table(agreement_rbf)
prop.table(table(agreement_rbf))