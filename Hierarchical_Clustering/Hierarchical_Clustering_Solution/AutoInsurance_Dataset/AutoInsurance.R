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
Auto <- read_csv(file.choose())

names(Auto)<-str_replace_all(names(Auto), c(" " = "_" , "," = "" ))


df_details <- function(df){
  feature_names <- names(df)
  length_feature <- length(feature_names)
  feature_row_count <- c()
  feature_null_check <- c()
  for (i in 1:length_feature){
    feature_row_count[i] <- sapply(Auto, function(x) length(unique(x)))[i]
    feature_null_check[i] <- is.null(df[i])
  }
  df <- data.frame("feature_names" = feature_names,
                   "feature_row_count" = feature_row_count,
                   "feature_null_check" = feature_null_check)
  return(df)
}

data_frame_output <- df_details(Auto)


df_filter <- Auto
str(Auto)

#label encoding, converting character to numeric data
df_filter$Customer	 <- as.numeric(as.factor(df_filter$Customer))
df_filter$State	 <- as.numeric(as.factor(df_filter$State))
df_filter$Response	 <- as.numeric(as.factor(df_filter$Response))
df_filter$Coverage	 <- as.numeric(as.factor(df_filter$Coverage))
df_filter$Education	 <- as.numeric(as.factor(df_filter$Education))
df_filter$Effective_To_Date	 <- as.numeric(as.factor(df_filter$Effective_To_Date))
df_filter$EmploymentStatus	 <- as.numeric(as.factor(df_filter$EmploymentStatus))
df_filter$Gender	 <- as.numeric(as.factor(df_filter$Gender))
df_filter$Location_Code	 <- as.numeric(as.factor(df_filter$Location_Code))
df_filter$Marital_Status	 <- as.numeric(as.factor(df_filter$Marital_Status))
df_filter$Policy_Type	 <- as.numeric(as.factor(df_filter$Policy_Type))
df_filter$Policy	 <- as.numeric(as.factor(df_filter$Policy))
df_filter$Renew_Offer_Type	 <- as.numeric(as.factor(df_filter$Renew_Offer_Type))
df_filter$Sales_Channel	 <- as.numeric(as.factor(df_filter$Sales_Channel))
df_filter$Vehicle_Class	 <- as.numeric(as.factor(df_filter$Vehicle_Class))
df_filter$Vehicle_Size	 <- as.numeric(as.factor(df_filter$Vehicle_Size))

boxplot(df_filter$Customer_Lifetime_Value)

nrow(unique(df_filter))
df_filter["Customer_Lifetime_Value"] <- Winsorize(df_filter$Customer_Lifetime_Value, probs = c(0.00, 0.90))
names(df_filter)

#rm(list=ls())

# Normalize the data
normalized_data <- scale(df_filter[,]) 

summary(normalized_data)

# Distance matrix
d <- dist(normalized_data, method = "euclidean") 

#hierarchical clustering for complete linkage
fit <- hclust(d, method = "complete")

# Display dendrogram
plot(fit) 
plot(fit, hang = -1)

groups <- cutree(fit, k = 5) # Cut tree into 6 clusters

rect.hclust(fit, k = 5, border = "red")

Group <- as.matrix(groups)

final <- data.frame(Group, df_filter)

aggregate(df_filter[,], by = list(final$Group), FUN = mean)

library(readr)
write_csv(final, "autoinsurance.csv")

getwd()

