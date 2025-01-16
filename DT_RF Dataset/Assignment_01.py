####################### ASSIGNMENT NO - 01 ##########################

'''Business Problem:
    A cloth manufacturing company is interested to know about the 
    different attributes contributing to high sales. Build a decision 
    tree & random forest model with Sales as target variable 
    (first convert it into categorical variable).
'''
    
# Data Gathering
import pandas as pd 
df=pd.read_csv("C:/Data Science/Assignment Data/DT_RF Dataset/Company_Data.csv")
df.head()

df.sample(5)

########################################################################

# EDA
df.info()
df.shape
df.isnull().sum()
df.duplicated().sum()


########################################################################

# Working with Numerical
import matplotlib.pyplot as plt 
import seaborn as sns 

# histogroam 
def col_histogram(col)  :  
    sns.histplot(df[col])

num=['Sales','CompPrice','Income','Advertising','Population','Price','Age','Education']
for i in num: 
    col_histogram(i)
    plt.show()
    
def col_displot(col) : 
    sns.displot(df, x=col, kind="kde")

for i in num: 
    col_displot(i)
    plt.show()
    
# Power transformer on given graphs 
#1) Income , Advertising  , Education

#outer ditation 
def box_plot(col): 
    sns.boxplot(df[col]) 
    
for i in num : 
    box_plot(i)
    plt.show()

# No outlier in data

####################################################################################

# Category

df.info()
cat=['ShelveLoc','Urban','US'] 
    
# heat map 
k=pd.crosstab(df['ShelveLoc'],df['Urban'])
k

k=pd.crosstab(df['ShelveLoc'],df['US'])
k    

k=pd.crosstab(df['Urban'],df['US'])
k

####################################################################################

# Preprocessing
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OrdinalEncoder 
lab= LabelEncoder()

set(df['ShelveLoc'])

ord=OrdinalEncoder(categories=[['Bad','Medium','Good']])

df['ShelveLoc']=ord.fit_transform(df[['ShelveLoc']])

df.sample(4)

df['ShelveLoc'].value_counts()


df['Urban']=lab.fit_transform(df['Urban'])
    
df['US']=lab.fit_transform(df['US'])

df

##################################################################################

# Spliting the data

from sklearn.model_selection import train_test_split 

X=df.iloc[:,:-1]
y=df[['US']]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)

########################################################################################

# Model Selection
# pip install xgboost
from sklearn.tree import DecisionTreeClassifier ,plot_tree 
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier
    
    
clf1=DecisionTreeClassifier()
clf2=RandomForestClassifier()
clf3=XGBClassifier()

clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
clf3.fit(X_train,y_train)

#prediction 
y_pre1=clf1.predict(X_test)
y_pre2=clf2.predict(X_test)
y_pre3=clf3.predict(X_test)


from sklearn.metrics import accuracy_score 
accuracy_score(y_test,y_pre1)
accuracy_score(y_test,y_pre2)
accuracy_score(y_test,y_pre3)

# for traing data 
y_train_p1=clf1.predict(X_train)
y_train_p2=clf2.predict(X_train)
y_train_p3=clf3.predict(X_train)

print(accuracy_score(y_train,y_train_p1))
print(accuracy_score(y_train,y_train_p2))
print(accuracy_score(y_train,y_train_p3))

# over fiting on data 

#Random forest is given good result 

####################################################################################

'''conclusion
After performing the Decision Tree and Random Forest models, I understand 
that a cloth manufacturing company is interested in analyzing sales. 
The objective is to build a predictive model with Sales as the target
 variable, focusing specifically on urban locations. 
 This approach helps the company make data-driven decisions to 
 optimize performance in key urban markets.'''



























    
    
