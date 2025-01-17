
# Assignment no: 08

##################################### Clustering Assignment ####################################

# 1.
'''Perform K means clustering on the airlines dataset to obtain optimum
 number of clusters. Draw the inferences from the clusters obtained. 
 Refer to EastWestAirlines.xlsx dataset.
'''

'''Business Objective:
    Perform clustering on EastWestAirlines based on similar charateristics.'''
# Clustering on Airline data
# Import important libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# Load the data
df = pd.read_csv("C:/Data Science/Assignment Data/EastWestAirlines(2).csv")
df.head() # display 1st five rows
df.columns

# Scatter Plot
plt.scatter(df['Bonus_miles'],df['Bonus_trans'])
plt.xlabel('Bonus_miles')
plt.ylabel('Bonus_trans')

# Initializes the KMeans clustering algorithm with 3 clusters
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Bonus_miles','Bonus_trans']])
y_predicted

# Assign Cluster Labels to Data 
df['cluster']=y_predicted   

df.head()

km.cluster_centers_


df1 = df[df.cluster==0] 
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1['Bonus_miles'],df1['Bonus_trans'],color='green')
plt.scatter(df2['Bonus_miles'],df2['Bonus_trans'],color='red')
plt.scatter(df3['Bonus_miles'],df3['Bonus_trans'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],
            color='white',marker='*',label="centroid")
plt.xlabel('Bonus_miles')
plt.ylabel('Bonus_trans')
plt.legend()

from sklearn.preprocessing import StandardScaler

# Check for missing values
print(df.isnull().sum())

# Select the features for clustering (numeric columns)
# Assuming columns like 'Balance', 'Bonus_miles', 'Flight_miles_12mo', etc., are suitable for clustering
features = df[['Balance', 'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12']]

# Normalize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

df.head()

# Optimal Number of clusters (Using Elbow curve method)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Determine the inertia for a range of cluster values
inertia = []
K = range(1, 11)  # Trying cluster numbers from 1 to 10
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the elbow method graph
plt.figure(figsize=(8,5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

#Apply K-means Clustering with the Chosen Number of Clusters
# Apply K-means with the optimal number of clusters
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Add the cluster assignments back to the dataset
print(df.head())


# Analyze the clusters
# Group by clusters and calculate the mean of each feature within each cluster
cluster_analysis = df.groupby('Cluster').mean()
print(cluster_analysis)

# Inferences from clusters
'''Cluster 0: May represent frequent flyers with high bonus miles and transactions.
Cluster 1: Could represent customers who have low travel frequency.
Cluster 2: Might be casual flyers with moderate use of the airline.
Cluster 3: May include customers who travel often but have fewer bonus miles.
'''

##########################################################################

# 2. 
'''Perform clustering for the crime data and identify the number of 
clusters formed and draw inferences. Refer to crime_data.csv dataset.
'''
'''Business Objective:
    Perform clustering on crime_data based on similar charateristics.'''
# Crime Dataset
# Clustering on Crime data
# Import important libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# Load the data
crime= pd.read_csv("C:/Data Science/Assignment Data/crime_data.csv")
crime.head() # display 1st five rows
crime.columns

#Data Preprocessing
# Removing useless column which is unnamed in the crime dataset
crime = crime.loc[:, ~crime.columns.str.contains('^Unnamed')]
crime

# Scatter Plot
plt.scatter(crime['UrbanPop'],crime['Assault'])
plt.xlabel('UrbanPop')
plt.ylabel('Assault')

# Initializes the KMeans clustering algorithm with 3 clusters
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(crime[['UrbanPop','Assault']])
y_predicted

# Assign Cluster Labels to Data 
crime['cluster']=y_predicted   

crime.head()

km.cluster_centers_


crime1 = crime[crime.cluster==0] 
crime2 = crime[crime.cluster==1]
crime3 = crime[crime.cluster==2]
plt.scatter(crime1['UrbanPop'],crime1['Assault'],color='green')
plt.scatter(crime2['UrbanPop'],crime2['Assault'],color='red')
plt.scatter(crime3['UrbanPop'],crime3['Assault'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],
            color='pink',marker='*',label="centroid")
plt.xlabel('UrbanPop')
plt.ylabel('Assault')
plt.legend()

from sklearn.preprocessing import StandardScaler

# Check for missing values
print(crime.isnull().sum())

# Select the features for clustering (numeric columns)
# Assuming columns like 'Balance', 'Bonus_miles', 'Flight_miles_12mo', etc., are suitable for clustering
features = crime[['UrbanPop', 'Assault', 'UrbanPop', 'Rape']]

# Normalize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

crime.head()

# Optimal Number of clusters (Using Elbow curve method)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Determine the inertia for a range of cluster values
inertia = []
K = range(1, 11)  # Trying cluster numbers from 1 to 10
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the elbow method graph
# The elbow point suggests the optimal number of clusters.
plt.figure(figsize=(8,5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

#Apply K-means Clustering with the Chosen Number of Clusters
# Apply K-means with the optimal number of clusters
kmeans = KMeans(n_clusters=4, random_state=42)
crime['Cluster'] = kmeans.fit_predict(scaled_features)

# Add the cluster assignments back to the dataset
print(crime.head())


# Analyze the clusters
# Group by clusters and calculate the mean of each feature within each cluster
cluster_analysis = crime.groupby('Cluster').mean()
print(cluster_analysis)
###########################################################################

# 3.
'''Analyze the information given in the following ‘Insurance Policy
 dataset’ to create clusters of persons falling in the same type. 
 Refer to Insurance Dataset.csv'''
 
'''Business Objective:
    Perform clustering on Insurance Dataset based on similar charateristics.'''
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# Load the data
insurance= pd.read_csv("C:/Data Science/Assignment Data/Insurance Dataset.csv")
insurance.head() # display 1st five rows
insurance.columns

#Data Preprocessing
# Removing useless column which is unnamed in the crime dataset
insurance = insurance.loc[:, ~insurance.columns.str.contains('^Unnamed')]
insurance

# Scatter Plot
plt.scatter(insurance['Age'],insurance['Income'])
plt.xlabel('Age')
plt.ylabel('Income')

# Initializes the KMeans clustering algorithm with 3 clusters
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(insurance[['Age','Income']])
y_predicted

# Assign Cluster Labels to Data 
insurance['cluster']=y_predicted   

insurance.head()

km.cluster_centers_


insurance1 = insurance[insurance.cluster==0] 
insurance2 = insurance[insurance.cluster==1]
insurance3 = insurance[insurance.cluster==2]
plt.scatter(insurance1['Age'],insurance1['Income'],color='green')
plt.scatter(insurance2['Age'],insurance2['Income'],color='red')
plt.scatter(insurance3['Age'],insurance3['Income'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],
            color='pink',marker='*',label="centroid")
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()

from sklearn.preprocessing import StandardScaler

# Check for missing values
print(crime.isnull().sum())

# Select the features for clustering (numeric columns)
# Assuming columns like 'Balance', 'Bonus_miles', 'Flight_miles_12mo', etc., are suitable for clustering
features =insurance[['Age', 'Premiums', 'Days to Renew', 'Claims made','Income']]

# Normalize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

insurance.head()

# Optimal Number of clusters (Using Elbow curve method)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Determine the inertia for a range of cluster values
inertia = []
K = range(1, 11)  # Trying cluster numbers from 1 to 10
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the elbow method graph
# The elbow point suggests the optimal number of clusters.
plt.figure(figsize=(8,5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

#Apply K-means Clustering with the Chosen Number of Clusters
# Apply K-means with the optimal number of clusters
kmeans = KMeans(n_clusters=4, random_state=42)
insurance['Cluster'] = kmeans.fit_predict(scaled_features)

# Add the cluster assignments back to the dataset
print(insurance.head())


# Analyze the clusters
# Group by clusters and calculate the mean of each feature within each cluster
cluster_analysis = insurance.groupby('Cluster').mean()
print(cluster_analysis)

###############################################################################
# 4.
'''Perform clustering analysis on the telecom dataset. The data is a mixture of both categorical and 
numerical data. It consists of the number of customers who churn. Derive insights and get possible 
information on factors that may affect the churn decision. Refer to Telco_customer_churn.xlsx dataset.
'''
'''Business Objective:
    Perform clustering on Telco_customer_churn based on similar charateristics.'''
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# Load the data
tel= pd.read_csv("C:/Data Science/Assignment Data/Telco_customer_churn.csv")
tel.head() # display 1st five rows
tel.columns

#Data Preprocessing
# Removing useless column which is unnamed in the crime dataset
#insurance = insurance.loc[:, ~insurance.columns.str.contains('^Unnamed')]
#insurance

# Scatter Plot
plt.scatter(tel['Avg Monthly GB Download'],tel['Avg Monthly Long Distance Charges'])
plt.xlabel('Avg Monthly GB Download')
plt.ylabel('Avg Monthly Long Distance Charges')

# Initializes the KMeans clustering algorithm with 3 clusters
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(tel[['Avg Monthly GB Download','Avg Monthly Long Distance Charges']])
y_predicted

# Assign Cluster Labels to Data 
tel['cluster']=y_predicted   

tel.head()

km.cluster_centers_


tel1 = tel[tel.cluster==0] 
tel2 = tel[tel.cluster==1]
tel3 = tel[tel.cluster==2]
plt.scatter(tel1['Avg Monthly GB Download'],tel1['Avg Monthly Long Distance Charges'],color='green')
plt.scatter(tel2['Avg Monthly GB Download'],tel2['Avg Monthly Long Distance Charges'],color='red')
plt.scatter(tel3['Avg Monthly GB Download'],tel3['Avg Monthly Long Distance Charges'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],
            color='pink',marker='*',label="centroid")
plt.xlabel('Avg Monthly GB Download')
plt.ylabel('Avg Monthly Long Distance Charges')
plt.legend()

from sklearn.preprocessing import StandardScaler

# Check for missing values
print(crime.isnull().sum())

# Select the features for clustering (numeric columns)
# Assuming columns like 'Balance', 'Bonus_miles', 'Flight_miles_12mo', etc., are suitable for clustering
features =tel[['Age', 'Premiums', 'Days to Renew', 'Claims made','Income']]

# Normalize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

tel.head()

# Optimal Number of clusters (Using Elbow curve method)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Determine the inertia for a range of cluster values
inertia = []
K = range(1, 11)  # Trying cluster numbers from 1 to 10
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the elbow method graph
# The elbow point suggests the optimal number of clusters.
plt.figure(figsize=(8,5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

#Apply K-means Clustering with the Chosen Number of Clusters
# Apply K-means with the optimal number of clusters
kmeans = KMeans(n_clusters=4, random_state=42)
tel['Cluster'] = kmeans.fit_predict(scaled_features)

# Add the cluster assignments back to the dataset
print(tel.head())


# Analyze the clusters
# Group by clusters and calculate the mean of each feature within each cluster
cluster_analysis = tel.groupby('Cluster').mean()
print(cluster_analysis)

###############################################################################

# 5.
'''Perform clustering on mixed data. Convert the categorical variables to 
numeric by using dummies or label encoding and perform normalization techniques. 
The dataset has the details of customers related to their auto insurance. 
Refer to Autoinsurance.csv dataset.
'''
#business objective
'''
business objective is to perform clustering on customers
based on their similar characteristics
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


autoIns = pd.read_csv("C:/Data Science/Assignment Data/Insurance Dataset.csv")
autoIns.columns
'''Index(['Premiums Paid', 'Age', 'Days to Renew', 'Claims made', 'Income'],
 dtype='object')'''

autoIns.dtypes

#most of colmns are of object type so we need to convert 
# them to numeric using dummies

autoIns
autoIns.describe()

# PDF and CDF
counts, bin_edges = np.histogram(autoIns['Claims made'], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

#Boxplot and outlier treatment

sns.boxplot(autoIns['Claims made'])
sns.boxplot(autoIns['Premiums Paid'])
sns.boxplot(autoIns['Age'])
sns.boxplot(autoIns['Days to Renew'])
sns.boxplot(autoIns['Income'])

#do not have outliers

#we need to remove outliers from other cols
#1
iqr = autoIns['Claims made'].quantile(0.75)-autoIns['Claims made'].quantile(0.25)
iqr

q1 = autoIns['Claims made'].quantile(0.25)
q3 = autoIns['Claims made'].quantile(0.75)

l_limit = q1-(1.5*iqr)
u_limit = q3+(1.5*iqr)

autoIns['Claims made'] = np.where(autoIns['Claims made'] >u_limit,u_limit,np.where(autoIns['Claims made']<l_limit,l_limit,autoIns['Claims made']))
sns.boxplot(autoIns['Claims made'])

#2
iqr = autoIns['Premiums Paid'].quantile(0.75)-autoIns['Premiums Paid'].quantile(0.25)
iqr

q1 = autoIns['Premiums Paid'].quantile(0.25)
q3 = autoIns['Premiums Paid'].quantile(0.75)

l_limit = q1-(1.5*iqr)
u_limit = q3+(1.5*iqr)

autoIns['Days to Renew'] = np.where(autoIns['Days to Renew'] >u_limit,u_limit,np.where(autoIns['Days to Renew']<l_limit,l_limit,autoIns['Days to Renew']))
sns.boxplot(autoIns['Days to Renew'])


q1 = autoIns['Premiums Paid'].quantile(0.25)
q3 = autoIns['Premiums Paid'].quantile(0.75)

l_limit = q1-(1.5*iqr)
u_limit = q3+(1.5*iqr)


autoIns.describe()


df_n= pd.get_dummies(autoIns)
df_n.shape

#now we have dataset df_n with all dtypes int
df_n.describe() 
#there is huge difference between min, max,and mean values in dataset cols
#so we need to normalize this data

def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

df_normal = norm_func(df_n)
desc = df_normal.describe()
desc

df_normal.columns
#in this Premiums Paid contains NAN values so 
#we will drop it

df_normal.drop(['Premiums Paid'],axis =1, inplace=True)
#now all data is normalized
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(df_normal,method='complete',metric='euclidean')
plt.figure(figsize=(15,8))
plt.title('Hierarchical clustering dendrogram')
plt.xlabel('index')
plt.ylabel('Distance')
#ref of dendrogram

sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()

#now apply clustering 
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters=3,
                                     linkage='complete',
                                     affinity='euclidean').fit(df_normal)
#apply labels to clusters
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
#assign this series to autoIns dataframe as column
autoIns['cluster'] = cluster_labels

autoInsNew = autoIns.iloc[:]
autoInsNew.iloc[:,2:].groupby(autoInsNew.cluster).mean()

autoInsNew.to_csv("AutoInsuranceNew.csv",encoding='utf-8')
import os
os.getcwd()

#KMeans Clustering on auto insurance
#for this we will used normalized data set df_normal

from sklearn.cluster import KMeans
#total sum of squares
TWSS = []

#initially we will find the ideal cluster number using elbow curve

k = list(range(2,8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_normal)
    TWSS.append(kmeans.inertia_)
  
TWSS
#benifits of client should written at the end of the code