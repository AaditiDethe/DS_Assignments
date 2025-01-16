
######################### ASSIGNMENT NO - 04 ##############################
'''
Objective: 

In the recruitment domain, HR departments face the significant 
challenge of verifying candidates' claims regarding their current or 
previous salaries, as these details heavily influence the salary 
expectations for the new role. Candidates may misrepresent their 
salary to secure a higher offer, leading to an inflated compensation 
structure, potential team imbalances, and an inefficient recruitment 
process. A data-driven approach to predict and verify salary claims 
can help recruiters make more informed decisions and minimize the risk 
of overpayment or recruitment misalignment.
'''

'''
Goal:
    
To build a predictive model using Decision Tree and Random Forest algorithms that estimates a candidate’s monthly income based on key variables'''

# Data  Collection
import pandas as pd 
import numpy as np 
df= pd.read_csv("C:/Data Science/Assignment Data/DT_RF Dataset\HR_DT.csv")

df.sample(4)
df.shape
set(df['Position of the employee'])
df['Position of the employee'].value_counts()
df.describe()
df.shape
df.info()
df.isnull().sum()
df.duplicated().sum()

# duplcate data is present in given data 
df=df.drop_duplicates()
df.duplicated().sum()
df.shape


################################################
# EDA
df['Position of the employee'].value_counts()
df['Position of the employee'].value_counts().plot(kind='bar')
import seaborn as sns 
import matplotlib.pyplot as plt 
sns.distplot(df[' monthly income of employee'])


###################################################
# Preprocessing of data
df.info()


# one hot encoding
from sklearn.preprocessing import OneHotEncoder 
one =OneHotEncoder()

m=pd.get_dummies(df['Position of the employee'],dtype=int)
df.shape
m.shape

combined_columns = pd.concat([df,m], axis=1)
combined_columns.sample(3)
combined_columns.shape
combined_columns.isnull().sum()

# Standigintion
from sklearn.preprocessing import StandardScaler,MinMaxScaler
min=MinMaxScaler()
#min.fit_transform(combined_columns['S'] no need 

####################################################
# Split the data
X=combined_columns.iloc[:,[1,3,4,5,6,7,8,9,10,11,12]]
y=combined_columns[' monthly income of employee']
X.shape

from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


###################################################
# Model selection
from sklearn.ensemble import RandomForestRegressor 
reg1=RandomForestRegressor() 

from sklearn.tree import DecisionTreeRegressor 
from sklearn.linear_model import LinearRegression,SGDRegressor,Lasso 

d_reg=DecisionTreeRegressor()
li_reg=LinearRegression()
las_reg =Lasso()

reg1.fit(X_train,y_train)

y_pre1=reg1.predict(X_test)
from sklearn.metrics import r2_score,mean_absolute_error
r2_score(y_test,y_pre1)

mean_absolute_error(y_test,y_pre1)
d_reg.fit(X_train,y_train)

y_pre2=d_reg.predict(X_test)
from sklearn.metrics import r2_score,mean_absolute_error
r2_score(y_test,y_pre2)

li_reg.fit(X_train,y_train)

y_pre3=li_reg.predict(X_test)
r2_score(y_test,y_pre3)

from sklearn.tree import plot_tree 
plot_tree(d_reg)
plt.show()

from sklearn.decomposition import PCA 
pca=PCA(n_components=2)

X_train_trf=pd.DataFrame(pca.fit_transform(X_train))
X_test_trf =pd.DataFrame(pca.fit_transform(X_test))  
d_reg.fit(X_train_trf,y_train)

y_pre3=d_reg.predict(X_test_trf)
r2_score(y_test,y_pre3)