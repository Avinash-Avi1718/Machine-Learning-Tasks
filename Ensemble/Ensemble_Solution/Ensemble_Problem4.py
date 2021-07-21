import pandas as pd
from sklearn.preprocessing import LabelEncoder
#from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

df = pd.read_excel("C:\\Users\\Avinash\\Desktop\\DataScience_AI\\Solutions\\Ensemble\\Ensemble_Password_Strength.xlsx")

# Dummy variables
df.head()
df.info()

def find_number (series) :
    result = []
    for record in series:
        res = any(chr.isdigit() for chr in str(record))
        result.append(res)
    return result

def uppercase_checker (series) :
    result = []
    for record in series:
        res = any(chr.isupper() for chr in str(record))
        result.append(res)
    return result

def lowercase_checker (series) :
    result = []
    for record in series:
        res = any(chr.islower() for chr in str(record))
        result.append(res)
    return result

import re

def spcl_ch_checker (series) :
    result = []
    string_check= re.compile('[@_!#$%^&*()<>?/\|}{~:-]')
    for record in series:
        res = any(string_check.search(chr) for chr in str(record))
        result.append(res)
    return result
    
number_check = find_number(df.characters[:])
upper_check = uppercase_checker(df.characters[:])
lower_check = lowercase_checker(df.characters[:])
special_ch_check = spcl_ch_checker(df.characters[:])

df['number_check'] = number_check
df['upper_check'] = upper_check
df['lower_check'] = lower_check
df['special_ch_check'] = special_ch_check

df.number_check = LabelEncoder().fit_transform(df.number_check)
df.upper_check = LabelEncoder().fit_transform(df.upper_check)
df.lower_check = LabelEncoder().fit_transform(df.lower_check)
df.special_ch_check = LabelEncoder().fit_transform(df.special_ch_check)

# Dummy variables
df.head()
df.info()

#df.columns = df.columns.str.replace(' ', '_')
# n-1 dummy variables will be created for n categories
#df = pd.get_dummies(df, columns = ["_Class_variable"], drop_first = True)


# Input and Output Split
pred_cols = [df.columns[2:6]]
predictors = df.iloc[:, 2:6]
type(predictors)

target = df["characters_strength"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)

#Applying Bagging

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 21)
knn.fit(x_train, y_train)

from sklearn import tree
clftree = tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier


bag_clf = BaggingClassifier(base_estimator = clftree, n_estimators = 500,
                            bootstrap = True, n_jobs = 1, random_state = 42)

bag_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, bag_clf.predict(x_test))
accuracy_score(y_test, bag_clf.predict(x_test))

# Evaluation on Training Data
confusion_matrix(y_train, bag_clf.predict(x_train))
accuracy_score(y_train, bag_clf.predict(x_train))


#Applying AdaBoosting

from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)

ada_clf.fit(x_train, y_train)

# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test))

# Evaluation on Training Data
confusion_matrix(y_train, ada_clf.predict(x_train))
accuracy_score(y_train, ada_clf.predict(x_train))


#Applying Stacking

#from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
#from sklearn.preprocessing import LabelEncoder

train_x, train_y = x_train, y_train
test_x, test_y = x_test, y_test

# Create the ensemble's base learners and meta learner
# Append base learners to a list
base_learners = []

# KNN classifier model
knn = KNeighborsClassifier(n_neighbors=2)
base_learners.append(knn)

# Decision Tree Classifier model
dtr = DecisionTreeClassifier(max_depth=4, random_state=123456)
base_learners.append(dtr)

# Logistic Regression Model
nb = GaussianNB()
base_learners.append(nb)

# Multi Layered Perceptron classifier
#mlpc = MLPClassifier(hidden_layer_sizes =(100, ), solver='lbfgs', random_state=1)
#base_learners.append(mlpc)

# Meta model using Logistic Regression
meta_learner = LogisticRegression(solver='lbfgs')


# Create the training meta data

# Create variables to store meta data and the targets
meta_data = np.zeros((len(base_learners), len(train_x)))
meta_targets = np.zeros(len(train_x))

# Create the cross-validation folds
KF = KFold(n_splits = 5)
meta_index = 0
for train_indices, test_indices in KF.split(train_x):
    # Train each learner on the K-1 folds and create meta data for the Kth fold

    for i in range(len(base_learners)):
        learner = base_learners[i]

        learner.fit(train_x.iloc[train_indices], train_y.iloc[train_indices])
        predictions = learner.predict_proba(train_x.iloc[test_indices])[:,0]

        meta_data[i][meta_index:meta_index+len(test_indices)] = predictions

    meta_targets[meta_index:meta_index+len(test_indices)] = train_y.iloc[test_indices]
    meta_index += len(test_indices)

# Transpose the meta data to be fed into the meta learner
meta_data = meta_data.transpose()

# Create the meta data for the test set and evaluate the base learners
test_meta_data = np.zeros((len(base_learners), len(test_x)))
base_acc = []

for i in range(len(base_learners)):
    learner = base_learners[i]
    learner.fit(train_x, train_y)
    predictions = learner.predict_proba(test_x)[:,0]
    test_meta_data[i] = predictions

    acc = metrics.accuracy_score(test_y, learner.predict(test_x))
    base_acc.append(acc)
test_meta_data = test_meta_data.transpose()

# Fit the meta learner on the train set and evaluate it on the test set
meta_learner.fit(meta_data, meta_targets)
ensemble_predictions = meta_learner.predict(test_meta_data)

acc = metrics.accuracy_score(test_y, ensemble_predictions)

# Print the results
for i in range(len(base_learners)):
    learner = base_learners[i]

    print(f'{base_acc[i]:.2f} {learner.__class__.__name__}')
    
print(f'{acc:.2f} Ensemble')


# Applying voting
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
# Instantiate the voting classifier
voting = VotingClassifier([('KNN', knn),
                           ('DTR', dtr),
                           ('NB', nb)])

# Fit classifier with the training data
voting.fit(x_train, y_train)

# Predict the most voted class
hard_predictions = voting.predict(x_test)

# Accuracy of hard voting
print('Hard Voting:', accuracy_score(y_test, hard_predictions))


# Instantiate the voting classifier
voting = VotingClassifier([('KNN', knn),
                           ('DTR', dtr),
                           ('NB', nb)],
                            voting = 'soft')

# Fit classifier with the training data
voting.fit(x_train, y_train)
knn.fit(x_train, y_train)
dtr.fit(x_train, y_train)
nb.fit(x_train, y_train)

# Predict the most probable class
soft_predictions = voting.predict(x_test)

# Get the base learner predictions
predictions_4 = knn.predict(x_test)
predictions_5 = dtr.predict(x_test)
predictions_6 = nb.predict(x_test)

# Accuracies of base learners
print('L4:', accuracy_score(y_test, predictions_4))
print('L5:', accuracy_score(y_test, predictions_5))
print('L6:', accuracy_score(y_test, predictions_6))

# Accuracy of Soft voting
print('Soft Voting:', accuracy_score(y_test, soft_predictions))