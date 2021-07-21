import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# Loading the data set

salary_train_data = pd.read_csv("C:\\Users\\Avinash\\Desktop\\DataScience_AI\\Solutions\\Naive_Bayes\\SalaryData_Train.csv",encoding = "ISO-8859-1")
salary_test_data = pd.read_csv("C:\\Users\\Avinash\\Desktop\\DataScience_AI\\Solutions\\Naive_Bayes\\SalaryData_Test.csv",encoding = "ISO-8859-1")

salary_train_data.info()

#combining all the columns of train data into single
salary_train_data['combined_features'] = salary_train_data[salary_train_data.columns[0:]].apply(
                        lambda x: ' '.join(x.dropna().astype(str)),
                        axis=1
                    )
salary_train_data = salary_train_data.iloc[:,13:15]


#combining all the columns of test data into single
salary_test_data['combined_features'] = salary_test_data[salary_test_data.columns[0:]].apply(
                        lambda x: ' '.join(x.dropna().astype(str)),
                        axis=1
                    )
salary_test_data = salary_test_data.iloc[:,13:15]


#adding train and test to get total data
salary_all_data = pd.concat([salary_train_data, salary_test_data])

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of salary texts into word count matrix format - Bag of Words
emails_bow = CountVectorizer(analyzer = split_into_words).fit(salary_all_data.combined_features)


# Defining BOW for all messages
all_emails_matrix = emails_bow.transform(salary_all_data.combined_features)

# For training messages
train_emails_matrix = emails_bow.transform(salary_train_data.combined_features)


# For testing messages
test_emails_matrix = emails_bow.transform(salary_test_data.combined_features)

# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_emails_matrix)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_emails_matrix)
train_tfidf.shape # (row, column)

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_emails_matrix)
test_tfidf.shape #  (row, column)

# Preparing a naive bayes model on training data set 
from sklearn.naive_bayes import MultinomialNB as MB


# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, salary_train_data.Salary)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == salary_test_data.Salary)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, salary_test_data.Salary) 
pd.crosstab(test_pred_m, salary_test_data.Salary)

# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == salary_train_data.Salary)
accuracy_train_m


# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.

classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(train_tfidf, salary_train_data.Salary)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == salary_test_data.Salary)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, salary_test_data.Salary) 
pd.crosstab(test_pred_lap, salary_test_data.Salary)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == salary_train_data.Salary)
accuracy_train_lap