
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb


from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import cross_val_score


# # Data Preparation

# In[2]:

train = pd.read_csv("C:/Users/Daniel/Data Science/Workspace/Titanic/1. Original Data/train.csv")
test = pd.read_csv("C:/Users/Daniel/Data Science/Workspace/Titanic/1. Original Data/test.csv")

all_data = pd.concat((train.loc[:,'Pclass':], test.loc[:,'Pclass':]), ignore_index=True) # drops PassengerId implicitly


# In[3]:

# extract Title from Names
all_data['Title'] = all_data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

# a map of more aggregated titles
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }

# we map each title
all_data['Title'] = all_data.Title.map(Title_Dictionary)


# In[4]:

all_data = all_data.drop('Name', axis=1)
all_data = all_data.drop('Ticket', axis=1)


# In[5]:

# To check how many columns have missing values - this can be repeated to see the progress made
def show_missing():
    missing = all_data.columns[all_data.isnull().any()].tolist()
    return missing

# Looking at categorical values
def cat_exploration(column):
    return all_data[column].value_counts()

# Imputing the missing values
def cat_imputation(column, value):
    all_data.loc[all_data[column].isnull(),column] = value


# In[6]:

# Number of missing values in each column
all_data[show_missing()].isnull().sum() 


# In[7]:

all_data = all_data.drop('Cabin', axis=1)


# In[8]:

all_data = all_data.fillna({
    'Fare' : all_data.Fare.median(),
    'Embarked': all_data.Embarked.mode()[0]})


# In[9]:

all_data.groupby(['Sex','Pclass','Title']).median()


# In[10]:

all_data["Age"] = all_data.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))


# In[11]:

all_data[show_missing()].isnull().sum() 


# # Feature Creation

# In[12]:

# introducing a new feature : the size of families (including the passenger)
all_data['FamilySize'] = all_data['Parch'] + all_data['SibSp'] + 1

# introducing other features based on the family size
all_data['Singleton'] = all_data['FamilySize'].map(lambda s : 1 if s == 1 else 0)
all_data['SmallFamily'] = all_data['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
all_data['LargeFamily'] = all_data['FamilySize'].map(lambda s : 1 if 5<=s else 0)


# # Final Preparation

# In[13]:

# Dummify categorical features
all_data = pd.get_dummies(all_data)  #, drop_first=True)


# In[14]:

# Scale to [0, 1] 
all_data = all_data.apply(lambda x: x/x.max(),axis=0)


# In[15]:

all_data.head()


# In[16]:

# creating matrices for sklearn:
X_train = all_data[:train.shape[0]].copy() # .shape[0] --> number of rows
X_test = all_data[train.shape[0]:].copy()
y = train.Survived.copy()


# # Models

# ## Logistic Regression

# In[17]:

model_log = LogisticRegressionCV().fit(X_train, y)

#log_preds = model_log.predict_proba(X_test)[:, 1]

log_preds = model_log.predict(X_test)


# In[18]:

# cross_val_score(model_log, X_train, y, scoring='accuracy', cv=5).mean()


# ## RandomForestClassifier

# In[19]:

param_grid = [ { 'max_depth' : [3, 4, 5],
                 'n_estimators': [20, 75, 100, 200],
                 'criterion': ['gini','entropy']}]

clf = GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid, scoring= 'accuracy', cv=5)

clf.fit(X_train, y)

summary = pd.DataFrame(clf.cv_results_)
summary.sort(columns='rank_test_score', ascending=True)


# In[20]:

print('Best score: {}'.format(clf.best_score_))
print('Best parameters: {}'.format(clf.best_params_))


# In[21]:

# model_rf = RandomForestClassifier(n_estimators=75, max_depth=4, criterion='gini').fit(X_train, y) # Best result with 20/4/gini
rf_preds = clf.predict(X_test)


# In[22]:

# cross_val_score(clf, X_train, y, scoring='accuracy', cv=5).mean()


# ## GradientBoostingClassifier

# In[23]:

param_grid = [ {'subsample' : [0.75, 1],
                'n_estimators':[200, 300], 
                'max_depth':[3, 4], 
                'learning_rate':[0.07, 0.09, 0.1] }]

clf2 = GridSearchCV(GradientBoostingClassifier(), param_grid, scoring= 'accuracy', cv=5)

clf2.fit(X_train, y)

# summary = pd.DataFrame(clf2.cv_results_)
# summary.sort(columns='rank_test_score', ascending=True)

print('Best score: {}'.format(clf2.best_score_))
print('Best parameters: {}'.format(clf2.best_params_))

#model_gbc = GradientBoostingClassifier(subsample=0.75, learning_rate=0.09, max_depth=3, n_estimators=200).fit(X_train, y)
#gbc_preds = model_gbc.predict(X_test)

gbc_preds = clf2.predict(X_test)


# In[24]:

# cross_val_score(clf2, X_train, y, scoring='accuracy', cv=5).mean()


# # Ensemble

# In[28]:

preds = (log_preds + gbc_preds + rf_preds)/3
preds = preds.round().astype(int)


# # Solution to CSV

# In[29]:

solution = pd.DataFrame({"PassengerId":test.PassengerId, "Survived":preds})
solution.to_csv("submission.csv", index = False) 

