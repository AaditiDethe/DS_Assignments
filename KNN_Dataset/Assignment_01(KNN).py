########################### ASSIGNMENT NO - 01(KNN) #########################

'''
Business Problem:
    A glass manufacturing plant uses different earth elements to design 
    new glass materials based on customer requirements. For that, they 
    would like to automate the process of classification as it’s a tedious 
    job to manually classify them. Help the company achieve its objective 
    by correctly classifying the glass type based on the other features 
    using KNN algorithm.
    
1.1 Business Objective

The objective is to automate the classification of glass types in a manufacturing plant. The company wants to categorize glass materials based on various chemical properties (e.g., Na, Mg, Al, Si, etc.) to streamline the production process and meet customer specifications efficiently. This will reduce the manual workload and improve accuracy in meeting product requirements.

1.2 Constraints

Accuracy: The classification model should achieve high accuracy to ensure that the glass types are identified correctly, as incorrect classifications could lead to product defects or unmet customer expectations.

Data Quality: The model's performance will depend on the quality of the input data (chemical properties).
'''

import pandas as pd
import numpy as np
data=pd.read_csv("C:/Data Science/Assignment Data/KNN_Dataset/glass.csv")
data.head()

#Now will check shape of the data
data.shape

#data Dictionary 
"""
Ri=Continous
Ni=Continous
Mi=Continous
Al=Continous
Si=Continous
K=Continous
Ba=Continous
Fe=Continous
Type=Discrete
"""


#########################################################################

# Data Preprocessing :

#Data Preprocessing
#first we will check the datatype
data.dtypes 
#Here we can see all the values are of numerical type

#Check the null values
data.isna().sum() 
#from ouput we can see that there are zero null values

#checking the outliers
import seaborn as sns
sns.boxplot(data[["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"]])

#Now outlier Treatment using replacement technique

IQR=data.quantile(0.75)-data.quantile(0.25)
IQR
lower_limit=data.quantile(0.25)-1.5*IQR
upper_limit=data.quantile(0.75)+1.5*IQR
IQR,lower_limit,upper_limit

df_replaced=pd.DataFrame(np.where(data>upper_limit,upper_limit,np.where(data<lower_limit,lower_limit,data)))
sns.boxplot(df_replaced),df_replaced.shape
#now from boxplot we can see that outliers has been removed

######################################################################

# EDA
df_replaced.describe() #by this we came to know min ,max,std,mean 25%,75%,50%

#Univariate Analysis
import matplotlib.pyplot as plt
plt.hist(data['RI'], bins=5, color='skyblue', edgecolor='black')

##########################################################################

#Model Building
#First we will go for normalizing the data 
'''def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
norm_data=norm_func(df_replaced.iloc[:,:9])
norm_data'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(df_replaced.iloc[:,:9])
X



#after Normalizing we can see 7th column became nan also there the imprtance of that feature less it contains maximum of the 0 values
#So we will drop that column
#norm_data.drop(norm_data.columns[7],axis=1,inplace=True) 

#Let us now apply x as input and y asoutput
#X=np.array(norm_data.iloc[:,:])
##Since is wbcd norm we are alredy excluding output column ,hence all rows and 
y=np.array(data['Type'])

#NOw we will split the data into training and testing state
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#Now we will apply model
from sklearn.neighbors import KNeighborsClassifier 
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
pred

#Now we will wvaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred,y_test)) 
pd.crosstab(pred,y_test)

#let us try to select correct value of k
acc=[]
#Running KNN algorithm for k=3 to 50 in steps of 2
for i in range(3,50,2):
    neigh=KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train,y_train)
    pred=neigh.predict(X_test)
    train_acc=np.mean(neigh.predict(X_train)==y_train)
    test_acc=np.mean(neigh.predict(X_test)==y_test)
    acc.append([train_acc,test_acc])
    
import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    