# Load the dataset
# Install the required packages
install.packages("readr")
install.packages("plyr")
install.packages("DescTools")
install.packages("stringr")
install.packages("CatEncoders")
install.packages("dplyr")
install.packages("caret")
install.packages("cluster")

library(cluster)
library(CatEncoders)
library(stringr)
library(DescTools)
library(readr)
library(plyr)
library(dplyr)
library(caret)
library(readxl)

#Load the dataset
Telecom <- read_excel(file.choose(), sheet = 'Telco_Churn')

#To rename oor replace column names , removing space in between 
names(Telecom)<-str_replace_all(names(Telecom), c(" " = "_" , "," = "" ))

#Check the null values in the dataset
sum(is.na(Telecom))

#checking for duplicate records
nrow(Telecom)
nrow(unique(Telecom))
#there are no duplicates in the dataset

# Loading/taking only required columns for analysis
df_filter <- Telecom[,c(5,6,7,9,12,13,22,24,25,26,27,28,29,30)]

str(Telecom)

#label encoding, converting character to numeric data
df_filter$Offer <- as.numeric(as.factor(df_filter$Offer))
df_filter$Internet_Type <- as.numeric(as.factor(df_filter$Internet_Type))
df_filter$Contract <- as.numeric(as.factor(df_filter$Contract))
df_filter$Payment_Method <- as.numeric(as.factor(df_filter$Payment_Method))

nrow(unique(df_filter))
df_filter["Total_Revenue"] <- Winsorize(df_filter$Total_Revenue, probs = c(0.00, 0.90))
names(df_filter)
boxplot(df_filter["Number_of_Referrals"])
boxplot(df_filter["Offer"])
boxplot(df_filter[,3])
boxplot(df_filter[,4])
boxplot(df_filter[,5])
boxplot(df_filter[,6])
boxplot(df_filter[,7])
boxplot(df_filter[,8])
boxplot(df_filter[,9])
boxplot(df_filter[,10])
boxplot(df_filter[,11])
boxplot(df_filter[,12])
boxplot(df_filter[,13])
boxplot(df_filter[,14])

names(df_filter)
#rm(list=ls())

df_filter[,-1]

telco_cust_distance <- daisy(df_filter,metric = 'gower')

fit <- hclust(telco_cust_distance, method = "complete")

# Display dendrogram
plot(fit) 

#Points are hanging in air hence use hang =-1 to touch ground level i.e. zero
plot(fit, hang = -1)

#divide or cut dendrogram into 6 clusters
groups <- cutree(fit, k = 5)# Cut tree into 6 clusters

#Seperate the cluster based on color and pattern
rect.hclust(fit, k = 5, border ="red")

#Providing name to new column being added 
Group <- as.matrix(groups)

final <- data.frame(Group, df_filter)

#Applying aggregate function to the required columns in the dataset
aggregate(df_filter[,], by = list(final$Group), FUN = mean)

#Save/export as csv file
library(readr)
write_csv(final, "telcom_customer_churn.csv")

#To check the working directory
getwd()

