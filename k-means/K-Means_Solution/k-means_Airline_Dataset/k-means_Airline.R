# Load the dataset
# Install the required packages
install.packages("readxl")
install.packages("plyr")
install.packages("DescTools")
install.packages("animation")
library(readxl)
library(plyr)
library(DescTools)
library(animation)

input <- read_excel(file.choose(), sheet = 'data')

# renaming the columns
names(input)[names(input) == "Award?"] <- "Award"
names(input)[names(input) == "ID#"] <- "ID"

count(input, 'Days_since_enroll')
names(input)

# Loading/taking only required columns for analysis
#We have removed C1,C3,C5,C6 as it does not provide useful info and variance is very low
mydata <- input[ , c(2,4,7:11)]

# Outlier treatment for columns which have outliers
names(mydata)
mydata['Balance'] <- Winsorize(mydata$Balance, probs = c(0.00, 0.92))
boxplot(mydata$Balance)
boxplot(mydata$cc1_miles)

mydata['Bonus_miles'] <- Winsorize(mydata$Bonus_miles, probs = c(0.00, 0.92))
boxplot(mydata$Bonus_miles)

mydata['Bonus_trans'] <- Winsorize(mydata$Bonus_trans, probs = c(0.00, 0.92))
boxplot(mydata$Bonus_trans)

mydata['Flight_miles_12mo'] <- Winsorize(mydata$Flight_miles_12mo, probs = c(0.00, 0.85))
boxplot(mydata$Flight_miles_12mo)

mydata['Flight_trans_12'] <- Winsorize(mydata$Flight_trans_12, probs = c(0.00, 0.85))
boxplot(mydata$Flight_trans_12)

boxplot(mydata$Days_since_enroll)

# Normalize the data
normalized_data <- scale(mydata[])
summary(normalized_data)


km <- kmeans.ani(normalized_data, 3)
km$centers

# kmeans clustering
km <- kmeans(normalized_data, 3) 
str(km)

# Elbow curve to decide the k value
twss <- NULL
for (i in 2:8) {
  twss <- c(twss, kmeans(normalized_data, centers = i)$tot.withinss)
}
twss

# Look for an "elbow" in the scree plot
plot(2:8, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")


# 3 Cluster Solution
fit <- kmeans(normalized_data, 3) 
str(fit)
fit$cluster
final <- data.frame(fit$cluster, mydata) # Append cluster membership

aggregate(mydata[, 2:7], by = list(fit$cluster), FUN = mean)
