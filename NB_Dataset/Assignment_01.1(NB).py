
########################### ASSIGNMENT NO - 01(NB) ###########################

'''
Business Problem:
    problem statment : Prepare a classification model using the 
    Naive Bayes algorithm for the salary dataset. 
    Train and test datasets are given separately. 
    Use both for model building.
    
Business Objectives:

Maximize:

Model accuracy Precision (if false positives are costly) Recall (if false negatives are costly) F1-score (balance of precision and recall) Minimize:

Misclassification rate False positives (>
50K wrongly predicted) Overfitting Irrelevant features

Constraints :

Data Quality:Missing or inconsistent values in the dataset (e.g., nulls or unknown values in workclass or occupation).

Imbalanced Classes:It is possible that the dataset may have an imbalanced distribution between salaries above and below $50K, which can affect the model's performance.

Independence Assumption of Naive Bayes:Naive Bayes assumes that features are conditionally independent of one another, which may not hold true for socio-economic attributes in this dataset.

Dataset Separation:Ensuring proper usage of train and test datasets without mixing, as they are already provided separately.
'''

# Data Collection
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score

#defining the data set which is in separate format test and train
train_df=pd.read_csv("SalaryData_Train.csv")
train_df.head()

test_df=pd.read_csv("SalaryData_Test.csv")
test_df.head()

print(train_df.shape)
print(test_df.shape)

print(train_df.columns)

print(train_df.dtypes)

train_df.isna().sum() #checking the null values ,There are no null values in the datset

train_df.info()

train_df.describe()

test_df.info()

test_df.describe()

import matplotlib.pyplot as plt
plt.hist(train_df['Salary'])

test_df['maritalstatus'].value_counts()

train_df[train_df.duplicated()].shape

train_df[train_df.duplicated()]


Train =train_df.drop_duplicates()
Train

Train.isnull().sum().sum()
## there is no nan values in the Train Data set

test_df[test_df.duplicated()].shape

test_df[test_df.duplicated()]

Test=test_df.drop_duplicates()
Test

Test.isnull().sum().sum()
## there is no nan values in the Train Data set

Train['Salary'].value_counts()

Test['Salary'].value_counts()

pd.crosstab(Train['occupation'],Train['Salary'])

pd.crosstab(Train['workclass'],Train['Salary'])

pd.crosstab(Train['workclass'],Train['occupation'])

import seaborn as sns
sns.countplot(x='Salary',data= Train)
plt.xlabel('Salary')
plt.ylabel('count')
plt.show()
Train['Salary'].value_counts()


sns.countplot(x='Salary',data= Test)
plt.xlabel('Salary')
plt.ylabel('count')
plt.show()
Test['Salary'].value_counts()

pd.crosstab(Train['Salary'],Train['education']).mean().plot(kind='bar')


string_columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

##Preprocessing the data. As, there are categorical variables
number = LabelEncoder()
for i in string_columns:
        Train[i]= number.fit_transform(Train[i])
        Test[i]=number.fit_transform(Test[i])

Train

Test


colnames=Train.columns
colnames

# storing the values in x_train,y_train,x_test & y_test for spliting the data in train and test for analysis
x_train = Train[colnames[0:13]].values
y_train = Train[colnames[13]].values
x_test = Test[colnames[0:13]].values
y_test = Test[colnames[13]].values


##Normalmization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

x_train
x_test

y_train
y_test

x_train = norm_func(x_train)
x_test =  norm_func(x_test)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

M_model=MB()
train_pred_multi=M_model.fit(x_train,y_train).predict(x_train)
test_pred_multi=M_model.fit(x_train,y_train).predict(x_test)

train_acc_multi=np.mean(train_pred_multi==y_train)
train_acc_multi ## train accuracy 74.42

test_acc_multi=np.mean(test_pred_multi==y_test)
test_acc_multi ## test acuracy 75.15

test_acc_multi=np.mean(test_pred_multi==y_test)
test_acc_multi ## test acuracy 75.15

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, test_pred_multi)

#print the matrix
confusion_matrix











