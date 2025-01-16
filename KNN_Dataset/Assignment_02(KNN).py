########################### ASSIGNMENT NO - 02(KNN) #########################
'''
Business Problem:
    A National Zoopark in India is dealing with the problem of segregation of the animals based on the different attributes they have. Build a KNN model to automatically classify the animals. Explain any inferences you draw in the documentation.

Objectives: Automate Animal Classification: Develop a KNN-based model to classify animals based on specific attributes. This classification should help in organizing animals within the zoo according to their characteristics.

Enhance Zoo Management: Facilitate easier animal management by categorizing animals, which can help with enclosure arrangements, feeding schedules, and medical care.

Improve Visitor Experience: By grouping similar animals together, the zoo can create a more organized layout that enhances the educational and viewing experience for visitors.

Support Conservation Efforts: Grouping animals with similar ecological or biological attributes may aid in conservation planning, breeding programs, and habitat simulations.

Constraints:

Data Quality: The model's performance relies on accurate, consistent, and comprehensive data on animal attributes. Any data inaccuracies could affect classification results.

Model Sensitivity: KNN models can be sensitive to the choice of the k value, so it may require tuning to achieve optimal classification accuracy.
'''
import pandas as pd
import numpy as np
data = pd.read_csv("C:/Data Science/Assignment Data/KNN_Dataset/Zoo.csv")
data.head()

data.shape

#data Dictionary
"""
animal.name :Discrete (Categorical)
hair	    :Discrete (Binary)
feathers	:Discrete (Binary)
eggs	    :Discrete (Binary)
milk	    :Discrete (Binary)
airborne	:Discrete (Binary)
aquatic	    :Discrete (Binary)
predator	:Discrete (Binary)
toothed	    :Discrete (Binary)
backbone	:Discrete (Binary)
breathes	:Discrete (Binary)
venomous	:Discrete (Binary)
fins	    :Discrete (Binary)
legs	    :Discrete (Integer)
tail	    :Discrete (Binary)
domestic	:Discrete (Binary)
"""

data.dtypes

#Check the null values
data.isna().sum() 
# there are zero null values



#checking the outliers
import seaborn as sns
sns.boxplot(data[["hair","feathers","eggs","milk","airborne","aquatic","predator","toothed","backbone","breathes","venomous","fins","legs","tail","domestic","catsize"]])

#Now outlier Treatment using replacement technique
data=data.iloc[:,1:]
IQR=data.quantile(0.75)-data.quantile(0.25)
IQR
lower_limit=data.quantile(0.25)-1.5*IQR
upper_limit=data.quantile(0.75)+1.5*IQR
IQR,lower_limit,upper_limit

df_replaced=pd.DataFrame(np.where(data>upper_limit,upper_limit,np.where(data<lower_limit,lower_limit,data)))
sns.boxplot(df_replaced),df_replaced.shape
#now from boxplot we can see that outliers has been removed



# EDA
df_replaced.describe() #by this we came to know min ,max,std,mean 25%,75%,50%

#Model Building
#NOw let us apply x as input and y as output

X=np.array(df_replaced.iloc[:,1:17])
y=np.array(data['type'])

#NOw we will split the data into training and testing state
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#Now we will apply model
from sklearn.neighbors import KNeighborsClassifier 
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
pred

# Now we will wvaluate the model
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














