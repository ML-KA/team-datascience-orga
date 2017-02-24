# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 08:35:46 2017

@author: Minh Ha Duong
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log
import seaborn as sns
from sklearn.model_selection import StratifiedKFold

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

sns.set_style("whitegrid")

# Read in files
training_data = pd.read_csv("train.csv", index_col = 0, na_values = "null")
test_data = pd.read_csv("test.csv", index_col = 0, na_values = "null")

# Explore data
training_data.head()
training_data.describe()

# transform Embarked & Sex in numeric values
training_data.loc[training_data["Embarked"]=="C", "Embarked"] = 1
training_data.loc[training_data["Embarked"]=="Q", "Embarked"] = 2
training_data.loc[training_data["Embarked"]=="S", "Embarked"] = 3
training_data.loc[training_data["Sex"]=="male", "Sex"] = 1
training_data.loc[training_data["Sex"]=="female", "Sex"] = 0

test_data.loc[test_data["Embarked"]=="C", "Embarked"] = 1
test_data.loc[test_data["Embarked"]=="Q", "Embarked"] = 2
test_data.loc[test_data["Embarked"]=="S", "Embarked"] = 3
test_data.loc[test_data["Sex"]=="male", "Sex"] = 1
test_data.loc[test_data["Sex"]=="female", "Sex"] = 0

# drop name, ticket and cabin columns
training_data = training_data.drop('Name', 1)
training_data = training_data.drop('Ticket', 1)
training_data = training_data.drop('Cabin', 1)

test_data = test_data.drop('Name', 1)
test_data = test_data.drop('Ticket', 1)
test_data = test_data.drop('Cabin', 1)

# fill empty & NaN values with median
training_data = training_data.replace(r'\s+', np.nan, regex=True)
training_data.apply(lambda x: x.fillna(x.median()),axis=1)
training_data.loc[training_data["Age"].isnull(),'Age'] = training_data.Age.median()
training_data.loc[training_data["Embarked"].isnull(),'Embarked'] = training_data.Age.median()

test_data = test_data.replace(r'\s+', np.nan, regex=True)
test_data.apply(lambda x: x.fillna(x.median()),axis=1)
test_data.loc[test_data["Age"].isnull(),'Age'] = test_data.Age.median()
test_data.loc[test_data["Embarked"].isnull(),'Embarked'] = test_data.Age.median()
test_data.loc[test_data["Fare"].isnull(),'Fare'] = test_data.Age.median()

# plot data
for column in training_data.columns:
    plt.figure(figsize=(10,2))
    sns.distplot(training_data[column])

# log loss function
def calculate_score(labels, pred):
    """Calculate the score according to competition rules."""
    score = 0
    # Add 10**-7 to avoid math error of log(0)
    for l, p in zip(labels, pred):
        score += l * log(p + 10**-7) + (1 - l) * log(1 - p + 10**-7)
    return (-1) * (1. / len(labels)) * score

## Define features and prediction
target = "Survived"
xTrain = training_data.ix[:, training_data.columns.difference([target])]
yTrain = training_data.ix[:,target]
xTest = test_data.ix[:, test_data.columns.difference([target])]


# It's easier to work with numpy
train_x_orig = xTrain.as_matrix()
train_y_orig = yTrain.as_matrix()

# Shuffle data
perm = np.random.permutation(len(train_y_orig))
train_x_orig = train_x_orig[perm]
train_y_orig = train_y_orig[perm]

# Get classifiers
classifiers = [
        ('Logistic Regression (C=1)', LogisticRegression(C=1)),
        ('Logistic Regression (C=1000)', LogisticRegression(C=10000)),
        
        ('SVM, adj.', SVC(probability=True,
                          kernel="rbf",
                          C=2.8,
                          gamma=.0073,
                          cache_size=200)),
        ('SVM, linear', SVC(probability=True,
                            kernel="linear",
                            C=0.025,
                            cache_size=200)),
        ('k nn (k=3)', KNeighborsClassifier(3)),
        ('k nn (k=5)', KNeighborsClassifier(5)),
        ('k nn (k=7)', KNeighborsClassifier(7)),
        ('k nn (k=21)', KNeighborsClassifier(21)),
        ('Decision Tree', DecisionTreeClassifier(max_depth=5)),
        ('Random Forest', RandomForestClassifier(n_estimators=50, n_jobs=10)),
        ('Random Forest 2', RandomForestClassifier(max_depth=5,
                                                   n_estimators=10,
                                                   max_features=1,
                                                   n_jobs=10)),
        ('AdaBoost', AdaBoostClassifier()),
        ('Naive Bayes', GaussianNB()),
        ('Gradient Boosting', GradientBoostingClassifier()),
    ]

kf = StratifiedKFold(n_splits=5)
i = 0

# Find best classifier
for clf_name, clf in classifiers:
    print("-" * 80)
    print("Name: %s (%i)" % (clf_name, i))
    score_estimates = []
    for train_ids, val_ids in kf.split(train_x_orig,train_y_orig):
        # Split labeled data into training and validation
        train_x = train_x_orig[train_ids]
        train_y = train_y_orig[train_ids]
        val_x = train_x_orig[val_ids]
        val_y = train_y_orig[val_ids]

        # Train classifier
        clf.fit(train_x, train_y)

        # Estimate loss
        val_pred = clf.predict_proba(val_x)[:, 1]
        score_estimates.append(calculate_score(val_y, val_pred))
        print("Estimated score: %0.4f" % score_estimates[-1])
    print("Average estimated score: %0.4f" %
          np.array(score_estimates).mean())
    i += 1
print("#" * 80)

# Train classifier on complete data
# classifier[13] is best model here
clf_name, clf = classifiers[13]
print("Train %s on complete data" %
      (clf_name))
clf.fit(train_x_orig, train_y_orig)

# Predict and write output
test_predicted = clf.predict_proba(xTest)[:, 1]
test_predicted

# save prediction to file
predictionDF = pd.DataFrame(data=test_predicted,index=test_data.index.values)
predictionDF.index.name = 'PassengerId'
predictionDF.columns = ['Survived']
predictionDF.to_csv('titanic_submission.csv')