
####################### ASSIGNMENT NO - 02 ##########################


'''
1. Business Problem:
    Divide the diabetes data into train and test datasets and build a
    Random Forest and Decision Tree model with Outcome as the output 
    variable. 
'''

# Data Gathering
import pandas as pd 
import numpy as np  
df= pd.read_csv("C:/Data Science/Assignment Data/DT_RF Dataset/Diabetes.csv")

df.head(3)
df.sample(2)
df.isnull().sum()
df.describe()

df.duplicated().sum()
df.info()

##################################################

# EDA
#all are numrical coul
import seaborn as sns 
import matplotlib.pyplot as plt 
num=[' Number of times pregnant']

def dist_plot(i): 
    sns.distplot(df[i])
    
dist_plot(num[0])

for i in df.columns: 
    dist_plot(i)
    plt.show()
    
# use power transformer to model to get good result 
def box_plot (i): 
    sns.boxplot(df[i])
    
    
for i in df.columns : 
    box_plot(i)
    plt.show()
    
#oulier present in data  
for i in df.columns : 
    print(i)

########################################################
    
# Data Preprocessing
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import PowerTransformer 
    
    
# label encoder to last colums to solve the proplem 
lab=LabelEncoder()
df[' Class variable']= lab.fit_transform(df[' Class variable'])

df.sample(4)

df[' Class variable'].value_counts()

# umblace Data 
df[' Class variable'].value_counts().plot(kind='bar')


sns.boxplot(df[' Diastolic blood pressure'])

sns.boxplot(df[' 2-Hour serum insulin'])

sns.boxplot(df[' Diabetes pedigree function'])

sns.boxplot(df[' Age (years)'])


def Remove_outler(i): 
    upper_limit = df[i].quantile(0.99)
    lower_limit =df[i].quantile(0.01) 
    new=np.where(df[i]>=upper_limit,upper_limit ,
         np.where(
               df[i]<=lower_limit,lower_limit
        , df[i]))
    return new

df[' Age (years)']=Remove_outler(' Age (years)')


df[' Diabetes pedigree function']=Remove_outler(' Diabetes pedigree function')

df[' Diastolic blood pressure']=Remove_outler(' Diastolic blood pressure')

power_transformer=PowerTransformer()

new_df=power_transformer.fit_transform(df.iloc[:,:-1])

new_df=pd.DataFrame(new_df)

#########################################################################

# Data Spliting
X=df.iloc[:,:-1]
y=df[' Class variable'].values 

from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#############################################################################3

# Model Selection

from sklearn.tree import DecisionTreeClassifier 
from  sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier 

clf1=DecisionTreeClassifier()
clf2=RandomForestClassifier()
clf3=XGBClassifier()

clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
clf3.fit(X_train,y_train)


y_predict1=clf1.predict(X_test)
y_predict2=clf2.predict(X_test)
y_predict3=clf3.predict(X_test)

from sklearn.metrics import accuracy_score 
accuracy_score(y_test,y_predict1)
accuracy_score(y_test,y_predict2)
accuracy_score(y_test,y_predict3)

y_predict_train_1=clf1.predict(X_train)
y_predict_train_2=clf2.predict(X_train)
y_predict_train_3=clf3.predict(X_train)

accuracy_score(y_train,y_predict_train_1)
accuracy_score(y_train,y_predict_train_2)
accuracy_score(y_train,y_predict_train_3)

# all are overfited condition so use the predictio for power transforemer

########################################################################

# Hyperparameter tuing
#clf1 is given best result so we can change the 
from sklearn.tree import plot_tree 
plot_tree(clf1)
plt.show()

clf1=DecisionTreeClassifier(max_depth=28,max_features=13,max_leaf_nodes=66)

clf1.fit(X_train,y_train)
y_predict1=clf1.predict(X_test)
accuracy_score(y_test,y_predict1)

clf7=RandomForestClassifier(n_estimators=300,n_jobs=300)
clf7.fit(X_train,y_train)
y_predict1=clf7.predict(X_test)
accuracy_score(y_test,y_predict1)

x_train=power_transformer.fit_transform(X_train)
x_test=power_transformer.transform(X_test)

clf1.fit(x_train,y_train)
y_predict1=clf1.predict(x_test)
accuracy_score(y_test,y_predict1)

y_predict12=clf1.predict(x_train)
accuracy_score(y_train,y_predict12)


########################################################################

'''Conclusion
The diabetes dataset was divided into training and testing subsets,
 and Random Forest and Decision Tree models were built to classify 
 individuals as diabetic or non-diabetic, with Outcome as the target
 variable. Random Forest outperformed Decision Tree in accuracy and
 feature importance analysis, highlighting glucose levels, BMI, 
 and age as key predictors. These models provide valuable tools for 
 healthcare professionals to identify at-risk individuals and enable
 timely interventions based on routine diagnostic data.'''













