############################ ASSIGNMENT NO - 02(NB) ##################################

'''Business Problem:
    This dataset contains information of users in a social network. 
    This social network has several business clients which can post ads
    on it. One of the clients has a car company which has just launched a 
    luxury SUV for a ridiculous price. Build a Bernoulli Naïve Bayes model 
    using this dataset and classify which of the users of the social 
    network are going to purchase this luxury SUV. 1 implies that there 
    was a purchase and 0 implies there wasn’t a purchase.
    
    1.1. What is the business objective?
    1.1.1 This will help you bring those audiences to your website who are interested in cars.
    And there will be many of those who are planning to buy a car in the near future.

    1.1.2 Communicating with your target audience over social media is always a great way to build good market reputation.
    Try responding to people's automobile related queries on twitter and facebook pages to be their first choice when it comes to buying a car.

1.2 Are there any constraints?
    Not having a clear market or social media strategy may result in reduced benefits for your business.
    Additional resources may be needed to manage your online presence.
    Social media is immediate and needs daily monitoring.
    If you don't actively manage your social media presence, you may not see any real benefits.
    There is a risk of unwanted or inappropriate behaviour on your site including bullying and harassment.
    Greater exposure inline has the potential to attract risks.
    Risks can include negative feedback information, leaks or hacking.
    '''

'''
Data dictionary

#User ID: Integer type which is not contributory
#Gender : Object type need to be label encoding
#Age : Integer
#EstimatedSalary : Integer
#Purchased : Integer type
'''

#############################################################################

# Data Collection : 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

df= pd.read_csv("C:/Data Science/Assignment Data/NB_Dataset/NB_Car_Ad.csv")

df.head(3)

df.shape

df.isnull().sum()

df.info()

df.duplicated().sum()

df.describe()

############################################################################

# EDA

df.sample(4)
df=df.drop(columns=['User ID'])
df.sample(3)
sns.scatterplot(df,x='EstimatedSalary',y='Age',hue='Gender')
sns.scatterplot(df,x='EstimatedSalary',y='Age',hue='Purchased')

df.info()
# outlyer diection in given system and working 


sns.boxplot(df['Age'])
sns.boxplot(df['EstimatedSalary'])
sns.boxplot(df['Purchased'])

df['Purchased'].value_counts()

df['Purchased'].value_counts().plot(kind="bar")

##############################################################################

# Data Preprocessing
#ordinal encoder 
from sklearn.preprocessing import LabelEncoder 

ord=LabelEncoder()
df['Gender']=ord.fit_transform(df['Gender'])

df.sample(3)

# scaler producting in data 
from sklearn.preprocessing import MinMaxScaler 

min=MinMaxScaler()

df['EstimatedSalary']=min.fit_transform(df[['EstimatedSalary']])


df['Age']=min.fit_transform(df[['Age']])

df.head(4)


df.sample(3)

##############################################################################

# Model Spliting
X=df.iloc[:,:-1]
y=df['Purchased'].values

from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test =train_test_split(X,y,random_state=42,test_size=0.2)


# Model Training
from sklearn.naive_bayes import MultinomialNB 
from sklearn.naive_bayes import BernoulliNB 


clf1 =MultinomialNB()
clf2=BernoulliNB()

clf1.fit(X_train,y_train)

clf2.fit(X_train,y_train)

y_pre1=clf1.predict(X_test)
y_pre2=clf2.predict(X_test)


# Accuracy
from sklearn.metrics import accuracy_score,f1_score 
accuracy_score(y_test,y_pre1)
accuracy_score(y_test,y_pre2)


from sklearn.ensemble import RandomForestClassifier 
clf3=RandomForestClassifier()
clf3.fit(X_train,y_train)


y_pre3=clf3.predict(X_test)
accuracy_score(y_test,y_pre3)

from xgboost import XGBClassifier


clf4=XGBClassifier()


clf4.fit(X_train,y_train)


y_pre4=clf4.predict(X_test)
accuracy_score(y_test,y_pre4)
X_train=X_train.iloc[:,1:]
X_test=X_test.iloc[:,1:]


clf3.fit(X_train,y_train)


y_pre3=clf3.predict(X_test)
accuracy_score(y_test,y_pre3)

clf4=XGBClassifier()
clf4.fit(X_train,y_train)
y_pre4=clf4.predict(X_test)
accuracy_score(y_test,y_pre4)