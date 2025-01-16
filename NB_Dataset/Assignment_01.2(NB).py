########################### ASSIGNMENT NO - 01.2(NB) #######################################3

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

df=pd.read_csv("SalaryData_Train.csv")

df.sample(4)

df.shape

test_df=pd.read_csv("C:/Data Science/Assignment Data/NB_Dataset/SalaryData_Test.csv")

test_df.sample(4)

test_df.shape

df.info()

def set_of_data(col) : 
    return len(set(df[col])) 

for i in df.columns: 
    print(f"{i} ----> {set_of_data(i)}")
    
# target colums is Salary 
sns.scatterplot(df,x='age',y='capitalgain',hue='Salary')

sns.scatterplot(df,x='age',y='capitalloss',hue='Salary')