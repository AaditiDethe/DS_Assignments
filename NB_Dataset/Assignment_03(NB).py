

########################### ASSIGNMENT NO - 03(NB) #########################
'''Business Problem:
    In this case study, you have been given Twitter data collected from an
    anonymous twitter handle. With the help of a Naïve Bayes model, predict 
    if a given tweet about a real disaster is real or fake.
1 = real tweet and 0 = fake tweet
'''

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

df=pd.read_csv("C:/Data Science/Assignment Data/NB_Dataset/Disaster_tweets_NB.csv")
df.sample(3)
df.info()

df.isnull().sum() #simpleImputer
df.shape
df.describe()

df.duplicated().sum()


#########################################################################

# EDA
df.info()
df.sample(3)

df['target'].value_counts()
df['text'][df['target']==1][0]
df['text'][df['target']==1][1]
df['text'][df['target']==1][3]
df['text'][df['target']==1][5]
df['text'][df['target']==1].shape
df['text'][df['target']==1]
df[df['target']==1].shape
df['text'][df['target']==1][7608]

df.info()
df.isnull().sum()
df.shape
df.sample(4)

# triming the values 


#########################################################################

# Preprocessing

df=df.dropna()

df.shape

df.isnull().sum()

df.sample(4)

X=df.iloc[:,:-1]
y=df['target']

# spliting of data 
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=42)

# vectorzing the word 

from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer

vector=CountVectorizer()
x_train_cv = vector.fit_transform(X_train)

x_test_cv=vector.fit_transform(X_test)

x_train_cv.shape
X_train.shape

#########################################################################

# Model Selection

from sklearn.naive_bayes import MultinomialNB 
clf1=MultinomialNB()

y_train

x_train_cv.shape


X=X.iloc[:,1:]

k=vector.fit_transform(X['keyword'])

numerical_df = pd.DataFrame(k.toarray(), columns=vector.get_feature_names_out())

numerical_df.shape

X.shape

k=vector.fit_transform(X['text'])

numerical_df1 = pd.DataFrame(k.toarray(), columns=vector.get_feature_names_out())

numerical_df1.shape

k=vector.fit_transform(X['location'])
numerical_df2 = pd.DataFrame(k.toarray(), columns=vector.get_feature_names_out())
numerical_df2.shape

new_df =pd.DataFrame()

combined = pd.concat([numerical_df, numerical_df1, numerical_df2], ignore_index=True)

combined.shape

combined_columns = pd.concat([numerical_df, numerical_df1, numerical_df2], axis=1)
combined_columns.shape


from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test =train_test_split(combined_columns,y,test_size=0.2,random_state=42)




from sklearn.naive_bayes import MultinomialNB 
clf1=MultinomialNB()

clf1.fit(X_train,y_train)

y_pre=clf1.predict(X_test)

from sklearn.metrics import accuracy_score 
accuracy_score(y_test,y_pre)




























