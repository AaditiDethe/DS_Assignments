
####################### ASSIGNMENT NO - 03 ##########################

'''
Business Problem:
    3.	Build a Decision Tree & Random Forest model on the fraud data. 
Treat those who have taxable_income <= 30000 as Risky and others as 
Good (discretize the taxable 
'''
# classfication of person salary is good or bad
import pandas as pd
df=pd.read_csv("C:/Data Science/Assignment Data/DT_RF Dataset/Fraud_check.csv")
df.sample(3)
df.shape

df['Taxable.Income'][df['Taxable.Income'] <=30000] =True
set(df['Marital.Status'])
set(df['Undergrad'])
set(df['Urban'])
df.sample(4)
df['Taxable.Income'][df['Taxable.Income'] >30000] =False  
df.sample(4)


import seaborn as sns 
import matplotlib.pyplot as plt 
df['Taxable.Income']=df['Taxable.Income'].replace(True,'Good')
df.sample(3)
df['Taxable.Income']=df['Taxable.Income'].replace(False,'Bad')
df.sample(5)

# Preprocessing
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
df.info()

df['Undergrad']=lab.fit_transform(df['Undergrad'])
df['Urban']=lab.fit_transform(df['Urban'])
df['Taxable.Income']=lab.fit_transform(df['Taxable.Income'])
df['Marital.Status']=lab.fit_transform(df['Marital.Status'])

sns.scatterplot(df,x='City.Population',y='Work.Experience',hue='Undergrad')

# city population is not import in given  table  
df=df.drop(columns=['City.Population'])
df['Taxable.Income'].value_counts()

#imbalance data 
X=df.iloc[:,[0,1,3,4]]
y=df['Taxable.Income'].values
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import plot_tree 
from sklearn.ensemble import RandomForestClassifier 

clf1=DecisionTreeClassifier()
clf2=RandomForestClassifier()

clf1.fit(X_train,y_train)

clf2.fit(X_train,y_train)

from sklearn.metrics import accuracy_score 
y_pre1=clf1.predict(X_test)
y_pre2=clf2.predict(X_test)  

accuracy_score(y_pre1,y_test)
accuracy_score(y_pre2,y_test)

# DecisionTreeClassifier ---> having more accuray then the random forest
# checking classfication report to anylis the overfiting condition in given data
from sklearn.metrics import classification_report ,f1_score,recall_score,precision_score
print(classification_report(y_test,y_pre1))
print(classification_report(y_test,y_pre2))

recall_score(y_test,y_pre1)
f1_score(y_test,y_pre1)












