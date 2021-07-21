import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# Loading the data set
car_data = pd.read_csv("C:\\Users\\Avinash\\Desktop\\DataScience_AI\\Solutions\\Naive_Bayes\\NB_Car_Ad.csv",encoding = "ISO-8859-1")
car_data.info()

#combining all the columns into single
car_data['combined_features'] = car_data[car_data.columns[1:4]].apply(
                        lambda x: ' '.join(x.dropna().astype(str)),
                        axis=1
                    )

car_data = car_data.iloc[:,4:6]

# CountVectorizer
# Convert a collection of text documents to a matrix of token counts
# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

car_train, car_test = train_test_split(car_data, test_size = 0.2)

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of email texts into word count matrix format - Bag of Words
cars_bow = CountVectorizer(analyzer = split_into_words).fit(car_data.combined_features)

# Defining BOW for all messages
all_emails_matrix = cars_bow.transform(car_data.combined_features)

# For training messages
train_cars_matrix = cars_bow.transform(car_train.combined_features)

# For testing messages
test_cars_matrix = cars_bow.transform(car_test.combined_features)

# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_emails_matrix)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_cars_matrix)
train_tfidf.shape # (row, column)

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_cars_matrix)
test_tfidf.shape #  (row, column)

# Preparing a naive bayes model on training data set 
from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, car_train.Purchased)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == car_test.Purchased)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, car_test.Purchased) 
pd.crosstab(test_pred_m, car_test.Purchased)

# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == car_train.Purchased)
accuracy_train_m

# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.

classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(train_tfidf, car_train.Purchased)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == car_test.Purchased)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, car_test.Purchased) 

pd.crosstab(test_pred_lap, car_test.Purchased)

# Training Data accuracy

train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == car_train.Purchased)
accuracy_train_lap