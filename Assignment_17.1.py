
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score

Url="https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv"
titanic = pd.read_csv(Url)
titanic.columns =['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
# "Sex" Coulumn has male/feamle as value. We can use LabelEncoder
# to convert these to int. male:1,female:0
lb = LabelEncoder()
titanic['Sex'] = lb.fit_transform(titanic['Sex'])
lb2 = LabelEncoder()
titanic['Embarked'] = lb2.fit_transform(titanic['Embarked'].fillna('0'))
Y = titanic.Survived
#keeping only required columns
X = titanic.drop(['Name', 'Ticket', 'Cabin', 'PassengerId','Survived','Embarked'], axis=1)
im = Imputer()
predictors = im.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(predictors, Y, random_state=1)
classifier=DecisionTreeClassifier()
classifier=classifier.fit(X_train,y_train )
y_predict = classifier.predict(X_test)

#It gives accuracy score of 0.7623318385650224
accuracy_score(y_test, y_predict)

