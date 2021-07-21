# Load the dataset
# Install the required packages
install.packages("readxl")
install.packages("plyr")
install.packages("DescTools")
library(readxl)
library(plyr)
library(DescTools)

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

# Distance matrix
d <- dist(normalized_data, method = "euclidean") 
d

#Hierarchical clustering and complete linkage
fit <- hclust(d, method = "complete")

# Display dendrogram
plot(fit) 
#Points are hanging in air hence use hang =-1 to touch ground level i.e. zero
plot(fit, hang = -1)

#divide or cut dendrogram into 6 clusters
groups <- cutree(fit, k = 6) # Cut tree into 6 clusters

#Seperate the cluster based on color and pattern
rect.hclust(fit, k = 6, border = "red")

#Providing name to new column being added 
Group <- as.matrix(groups)

final <- data.frame(Group, mydata)

#Applying aggregate function to the required columns in the dataset
aggregate(mydata[, 1:7], by = list(final$Group), FUN = mean)

#Save/export as csv file
library(readr)
write_csv(final, "hclustoutput.csv")

#To check the working directory
getwd()

