# Load the dataset
# Install the required packages
installed.packages("readr")
install.packages("plyr")
install.packages("DescTools")
library(readr)
library(plyr)
library(DescTools)

#Load the dataset
crime <- read.csv(file.choose())
#To check the column names of the dataset
names(crime)

#To rename the unnamed column
names(crime)[names(crime) == "X"] <- "States"

#checking for null values in each column
is.null(crime['States'])
is.null(crime['Murder'])
is.null(crime['Assault'])
is.null(crime['UrbanPop'])
is.null(crime['Rape'])

#To check the null values in the whole dataset
sum(is.na(crime))

#checking for duplicate records
nrow(crime)
nrow(unique(crime))
#there are no duplicates in the dataset


# Loading/taking only required columns for analysis
crime1 <- crime[ , c(2:5)]

# Outlier treatment for columns which have outliers
names(crime1)
boxplot(crime1$Murder)
boxplot(crime1$Assault)
boxplot(crime1$UrbanPop)
boxplot(crime1$Rape)

#Winsorising the column Rape as it contains outliers
crime1['Rape'] <- Winsorize(crime1$Rape, probs = c(0.00, 0.92))
boxplot(crime1$Rape)


# Normalize the data
normalized_data <- scale(crime1[])
summary(normalized_data)

# Distance matrix
d <- dist(normalized_data, method = "euclidean") 

# Hierarchical clustering and applying complete linkage method
fit <- hclust(d, method = "complete")

# Display dendrogram
plot(fit) 

#As points are hanging provide hang =-1 to touch the axis at ground level
plot(fit, hang = -1)

#Based on dendrogram, let us plot 3 clusters
groups <- cutree(fit, k = 3) # Cut tree into 3 clusters

#Cluster into 3 based on colour and shape
rect.hclust(fit, k = 3, border = "red")

#Providing name to the newly adding column for cluster group
Group <- as.matrix(groups)

final <- data.frame(Group, crime1)

#Applying aggregate function to required columns to the dataset
aggregate(crime1[, 1:4], by = list(final$Group), FUN = mean)

library(readr)
write_csv(final, "hclustoutput.csv")

getwd()
