from __future__ import print_function
import pandas as pd
import numpy as np

#import libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import RepeatedKFold, cross_val_score, KFold

data = pd.read_csv("B:\python\Machine Learning\crop_recommendation_randomforest\Crop_recommendation.csv")
print(data.shape)
data.columns = data.columns.str.replace(' ', '')
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Displaying correlation between each features
fig, ax = plt.subplots(1, 1, figsize=(15, 9))
sns.heatmap(data.corr(), annot=True,cmap='viridis')
ax.set(xlabel='features')
ax.set(ylabel='features')
plt.title('Correlation between different features', fontsize = 15, c='black')
plt.show()

# Displaying each features in Histogram
data_elem= data[['N','P','K']]
data_cond= data[['temperature','humidity','ph','rainfall']]
for i in data_elem.columns:
    plt.hist(data_elem[i])
    plt.title(i)
    plt.show()
for i in data_cond.columns:
    plt.hist(data_cond[i])
    plt.title(i)
    plt.show()
    
# Displaying table between crops and features
pd.pivot_table(data,index='label',values=['temperature','rainfall','N','P','K','humidity','ph'])

# Making Machine Learning model
features = data[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = data['label']
acc = []
model = []

# Splitting dataset into training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features,target,test_size = 0.2,random_state = 2)

# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(x_train,y_train)

predicted_values = RF.predict(x_test)

x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('RF')
print("Random Forest Accuracy is: ", x)

print(classification_report(y_test,predicted_values))

# Cross validation score
score = cross_val_score(RF,features,target,cv=5)
print('Cross validation score: ',score)

#Print Train Accuracy
rf_train_accuracy = RF.score(x_train,y_train)
print("Training accuracy = ",RF.score(x_train,y_train))

#Print Test Accuracy
rf_test_accuracy = RF.score(x_test,y_test)
print("Testing accuracy = ",RF.score(x_test,y_test))


y_pred = RF.predict(x_test)
y_true = y_test

#  Using heatmap to see the correlation between actual and predicted value
from sklearn.metrics import confusion_matrix

cm_rf = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize=(15,10))
sns.heatmap(cm_rf, annot=True, linewidth=0.5, fmt=".0f",  cmap='viridis', ax = ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title('Predicted vs actual')
plt.show()

from sklearn.datasets import make_classification
from numpy import mean
from numpy import std
X, Y = make_classification(n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=1)
cv = KFold(n_splits=10, random_state=1, shuffle=True)

# Cross validation score
score = cross_val_score(RF, features, target, cv=cv, scoring="accuracy")
print('Cross validation score: ',score)
print('Accuracy: %.3f (%.3f)' % (mean(score), std(score)))