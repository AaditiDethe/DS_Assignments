
################################# PCA Assignment ################################
'''
Problem Statement: -
Perform hierarchical and K-means clustering on the dataset. After that, perform PCA on the 
dataset and extract the first 3 principal components and make a new dataset with these 3 
principal components as the columns. Now, on this new dataset, perform hierarchical and
 K-means clustering. Compare the results of clustering on the original dataset and clustering 
 on the principal components dataset (use the scree plot technique to obtain the optimum 
number of clusters in K-means clustering and check if you’re getting similar results with 
and without PCA).

'''
import pandas as pd
df=pd.read_csv("C:/PCA/heart disease.csv")
df.head()

df.columns

df.isnull().sum()

#Plotting boxplot
import seaborn as sns
sns.boxplot(df[['age', 'sex','cp']])

#Plotting boxplot
import seaborn as sns
sns.boxplot(df[['trestbps', 'chol', 'cp']])

#Plotting boxplot
import seaborn as sns
sns.boxplot(df[['restecg', 'thalach', 'exang']])

#Plotting boxplot
import seaborn as sns
sns.boxplot(df[['oldpeak', 'slope', 'ca']])

#Plotting boxplot
import seaborn as sns
sns.boxplot(df[['thal','target']])

outliers_columns=['trestbps','chol','fbs','thalach','oldpeak','ca','thal']

import numpy as np
# Apply log transformation to the specified columns
for column in outliers_columns:
    # Adding a small constant to avoid issues with log(0)
    df[column] = np.log1p(df[column])
    
# Display the transformed data
df.head()

def remove_out(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        #Remove outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df

df= remove_out(df,outliers_columns)

#Plotting boxplot
import seaborn as sns
sns.boxplot(df[['trestbps', 'chol', 'cp']])

#normalize the data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df_scaled=scaler.fit_transform(df)

df_scaled=pd.DataFrame(df_scaled, columns=df.columns)

sns.boxplot(df_scaled)

# Perform K-means clustering based on alcohol and malic acid
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(df_scaled[['age', 'trestbps']])

# Get cluster centers (centroids)
centroids = kmeans.cluster_centers_

# Create dataframes for different clusters
df_cluster1 = df_scaled[y_kmeans == 0]
df_cluster2 = df_scaled[y_kmeans == 1]
df_cluster3 = df_scaled[y_kmeans == 2]

# Plot the clusters and centroids
plt.scatter(df_cluster1['age'], df_cluster1['trestbps'], s=50, c='blue', label='Cluster 1')
plt.scatter(df_cluster2['age'], df_cluster2['trestbps'], s=50, c='red', label='Cluster 2')
plt.scatter(df_cluster3['age'], df_cluster3['trestbps'], s=50, c='yellow', label='Cluster 3')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black', label='Centroids')
plt.xlabel('age')
plt.ylabel('trestbps')
plt.title('K-means Clustering Heart(age vs trestbps)')
plt.legend()
plt.show()

# Perform K-means clustering based on alcohol and malic acid
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(df_scaled[['age', 'chol']])

# Get cluster centers (centroids)
centroids = kmeans.cluster_centers_

# Create dataframes for different clusters
df_cluster1 = df_scaled[y_kmeans == 0]
df_cluster2 = df_scaled[y_kmeans == 1]
df_cluster3 = df_scaled[y_kmeans == 2]

# Plot the clusters and centroids
plt.scatter(df_cluster1['age'], df_cluster1['chol'], s=50, c='blue', label='Cluster 1')
plt.scatter(df_cluster2['age'], df_cluster2['chol'], s=50, c='red', label='Cluster 2')
plt.scatter(df_cluster3['age'], df_cluster3['chol'], s=50, c='yellow', label='Cluster 3')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black', label='Centroids')
plt.xlabel('age')
plt.ylabel('chol')
plt.title('K-means Clustering Heart(age vs chol)')
plt.legend()
plt.show()

from sklearn.metrics import davies_bouldin_score

# Calculate Davies-Bouldin Index
db_index = davies_bouldin_score(df_scaled, y_kmeans)
print(f'Davies-Bouldin Index: {db_index}')

#calculate silhoutte score average
from sklearn.metrics import silhouette_score
sil_score=silhouette_score(df_scaled, y_kmeans)
print(sil_score)

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(df_scaled)
# Create a new DataFrame with the principal components
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
# Print the explained variance ratio for each component
print('Explained Variance Ratio:', pca.explained_variance_ratio_)
     

#perform kmeans clustering on PC1 and PC2
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(principal_df[['PC1', 'PC2']])
# Get cluster centers (centroids)
centroids = kmeans.cluster_centers_
# Create dataframes for different clusters
df_cluster1 = principal_df[y_kmeans == 0]
df_cluster2 = principal_df[y_kmeans == 1]
df_cluster3 = principal_df[y_kmeans == 2]
# Plot the clusters and centroids
plt.scatter(df_cluster1['PC1'], df_cluster1['PC2'], s=50, c='blue', label='Cluster 1')
plt.scatter(df_cluster2['PC1'], df_cluster2['PC2'], s=50, c='red', label='Cluster 2')
plt.scatter(df_cluster3['PC1'], df_cluster3['PC2'], s=50, c='yellow', label='Cluster 3')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black', label='Centroids')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('K-means Clustering of heart Dataset (PC1 vs PC2)')
plt.legend()
plt.show()

#perform kmeans clustering on PC1 and PC3
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(principal_df[['PC1', 'PC3']])
# Get cluster centers (centroids)
centroids = kmeans.cluster_centers_
# Create dataframes for different clusters
df_cluster1 = principal_df[y_kmeans == 0]
df_cluster2 = principal_df[y_kmeans == 1]
df_cluster3 = principal_df[y_kmeans == 2]
# Plot the clusters and centroids
plt.scatter(df_cluster1['PC1'], df_cluster1['PC3'], s=50, c='blue', label='Cluster 1')
plt.scatter(df_cluster2['PC1'], df_cluster2['PC3'], s=50, c='red', label='Cluster 2')
plt.scatter(df_cluster3['PC1'], df_cluster3['PC3'], s=50, c='yellow', label='Cluster 3')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black', label='Centroids')
plt.xlabel('PC1')
plt.ylabel('PC3')
plt.title('K-means Clustering of heart Dataset (PC1 vs PC3)')
plt.legend()
plt.show()

#perform kmeans clustering on PC2 and PC3
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(principal_df[['PC2', 'PC3']])
# Get cluster centers (centroids)
centroids = kmeans.cluster_centers_
# Create dataframes for different clusters
df_cluster1 = principal_df[y_kmeans == 0]
df_cluster2 = principal_df[y_kmeans == 1]
df_cluster3 = principal_df[y_kmeans == 2]
# Plot the clusters and centroids
plt.scatter(df_cluster1['PC2'], df_cluster1['PC3'], s=50, c='blue', label='Cluster 1')
plt.scatter(df_cluster2['PC2'], df_cluster2['PC3'], s=50, c='green', label='Cluster 2')
plt.scatter(df_cluster3['PC2'], df_cluster3['PC3'], s=50, c='red', label='Cluster 3')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black', label='Centroids')
plt.xlabel('PC2')
plt.ylabel('PC3')
plt.title('K-means Clustering of heart Dataset (PC2 vs PC3)')
plt.legend()
plt.show()

#calculate silhoutte score average
from sklearn.metrics import silhouette_score
sil_score=silhouette_score(principal_df, y_kmeans)
print(sil_score)

from sklearn.metrics import davies_bouldin_score

# Calculate Davies-Bouldin Index
db_index = davies_bouldin_score(df_scaled, y_kmeans)
print(f'Davies-Bouldin Index: {db_index}')

'''Business Objective:


The goal is to predict whether a patient has heart disease based on the medical attributes in the dataset. This can help healthcare providers identify at-risk patients and administer timely interventions.

Constraints:

Data Quality: The model's accuracy depends on the completeness and quality of the data. Missing values, noise, or outliers could impact predictions.

Interpretability: Medical predictions often need to be explainable for healthcare professionals to trust the model's outputs.'''

####################################################################################################
'''Business Objective:

The objective for this dataset could be to build a predictive model that classifies wines into different types based on their chemical properties. This model could help wine producers or distributors quickly and accurately categorize wines, ensuring quality control, consistency, and better marketing strategies.'''

import pandas as pd
df=pd.read_csv("C:/PCA/wine.csv")
df.head()
df.columns

df.isnull.sum()

df.describe()

#Plotting boxplot
import seaborn as sns
sns.boxplot(df[['Alcohol', 'Malic', 'Ash']])

sns.boxplot(df[['Phenols','Flavanoids', 'Nonflavanoids', 'Proanthocyanins']])

sns.boxplot(df[['Alcalinity', 'Magnesium']])

sns.boxplot(df[['Color', 'Hue', 'Dilution']])

sns.boxplot(df[['Proline']])

outliers_columns=['Malic', 'Ash', 'Alcalinity', 'Magnesium','Proanthocyanins', 'Color', 'Hue']

import numpy as np
# Apply log transformation to the specified columns
for column in outliers_columns:
    # Adding a small constant to avoid issues with log(0)
    df[column] = np.log1p(df[column])
    
df.head()

sns.boxplot(df[['Alcalinity', 'Magnesium']])

def remove_out(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        #Remove outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df

sns.boxplot(df[['Alcalinity', 'Magnesium']])
sns.boxplot(df[['Color', 'Hue', 'Dilution']])
sns.boxplot(df[['Alcohol', 'Malic', 'Ash']])

#normalize the data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df_scaled=scaler.fit_transform(df)

df_scaled=pd.DataFrame(df_scaled, columns=df.columns)

sns.boxplot(df_scaled)

#perform clustering based on alcohol content
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

#hierarchical clustering
Clust=AgglomerativeClustering(n_clusters=3)
y_clust=Clust.fit_predict(df_scaled)

#create dataframes for different clusters
df_cluster1=df[y_clust==0]
df_cluster2=df[y_clust==1]
df_cluster3=df[y_clust==2]

plt.scatter(df_cluster1['Alcohol'], df_cluster1['Malic'], s=50, c='blue', label='Cluster 1')
plt.scatter(df_cluster2['Alcohol'], df_cluster2['Malic'], s=50, c='red', label='Cluster 2')
plt.scatter(df_cluster3['Alcohol'], df_cluster3['Malic'], s=50, c='yellow', label='Cluster 3')

# Perform K-means clustering based on alcohol and malic acid
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(df_scaled[['Alcohol', 'Malic']])

# Get cluster centers (centroids)
centroids = kmeans.cluster_centers_

# Create dataframes for different clusters
df_cluster1 = df_scaled[y_kmeans == 0]
df_cluster2 = df_scaled[y_kmeans == 1]
df_cluster3 = df_scaled[y_kmeans == 2]

# Plot the clusters and centroids
plt.scatter(df_cluster1['Alcohol'], df_cluster1['Malic'], s=50, c='blue', label='Cluster 1')
plt.scatter(df_cluster2['Alcohol'], df_cluster2['Malic'], s=50, c='red', label='Cluster 2')
plt.scatter(df_cluster3['Alcohol'], df_cluster3['Malic'], s=50, c='yellow', label='Cluster 3')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black', label='Centroids')
plt.xlabel('Alcohol')
plt.ylabel('Malic Acid')
plt.title('K-means Clustering Wine(Alcohol vs Malic Acid)')
plt.legend()
plt.show()

# Perform K-means clustering based on alcohol and malic acid
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(df_scaled[['Flavanoids', 'Nonflavanoids']])

# Get cluster centers (centroids)
centroids = kmeans.cluster_centers_

# Create dataframes for different clusters
df_cluster1 = df_scaled[y_kmeans == 0]
df_cluster2 = df_scaled[y_kmeans == 1]
df_cluster3 = df_scaled[y_kmeans == 2]

# Plot the clusters and centroids
plt.scatter(df_cluster1['Flavanoids'], df_cluster1['Nonflavanoids'], s=50, c='blue', label='Cluster 1')
plt.scatter(df_cluster2['Flavanoids'], df_cluster2['Nonflavanoids'], s=50, c='red', label='Cluster 2')
plt.scatter(df_cluster3['Flavanoids'], df_cluster3['Nonflavanoids'], s=50, c='yellow', label='Cluster 3')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black', label='Centroids')
plt.xlabel('Flavanoids')
plt.ylabel('Magnesium')
plt.title('K-means Clustering of Wine Dataset (Flavanoids vs Nonflavanoids)')
plt.legend()
plt.show()

from sklearn.metrics import davies_bouldin_score

# Calculate Davies-Bouldin Index
db_index = davies_bouldin_score(df_scaled, y_kmeans)
print(f'Davies-Bouldin Index: {db_index}')

#calculate silhoutte score average
from sklearn.metrics import silhouette_score
sil_score=silhouette_score(df_scaled, y_kmeans)
print(sil_score)

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(df_scaled)
# Create a new DataFrame with the principal components
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
# Print the explained variance ratio for each component
print('Explained Variance Ratio:', pca.explained_variance_ratio_)

#perform kmeans clustering on PC1 and PC2
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(principal_df[['PC1', 'PC2']])
# Get cluster centers (centroids)
centroids = kmeans.cluster_centers_
# Create dataframes for different clusters
df_cluster1 = principal_df[y_kmeans == 0]
df_cluster2 = principal_df[y_kmeans == 1]
df_cluster3 = principal_df[y_kmeans == 2]
# Plot the clusters and centroids
plt.scatter(df_cluster1['PC1'], df_cluster1['PC2'], s=50, c='blue', label='Cluster 1')
plt.scatter(df_cluster2['PC1'], df_cluster2['PC2'], s=50, c='red', label='Cluster 2')
plt.scatter(df_cluster3['PC1'], df_cluster3['PC2'], s=50, c='yellow', label='Cluster 3')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black', label='Centroids')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('K-means Clustering of Wine Dataset (PC1 vs PC2)')
plt.legend()
plt.show()

#perform kmeans clustering on PC1 and PC3
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(principal_df[['PC1', 'PC3']])
# Get cluster centers (centroids)
centroids = kmeans.cluster_centers_
# Create dataframes for different clusters
df_cluster1 = principal_df[y_kmeans == 0]
df_cluster2 = principal_df[y_kmeans == 1]
df_cluster3 = principal_df[y_kmeans == 2]
# Plot the clusters and centroids
plt.scatter(df_cluster1['PC1'], df_cluster1['PC3'], s=50, c='blue', label='Cluster 1')
plt.scatter(df_cluster2['PC1'], df_cluster2['PC3'], s=50, c='red', label='Cluster 2')
plt.scatter(df_cluster3['PC1'], df_cluster3['PC3'], s=50, c='yellow', label='Cluster 3')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black', label='Centroids')
plt.xlabel('PC1')
plt.ylabel('PC3')
plt.title('K-means Clustering of Wine Dataset (PC1 vs PC3)')
plt.legend()
plt.show()

#perform kmeans clustering on PC2 and PC3
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(principal_df[['PC2', 'PC3']])
# Get cluster centers (centroids)
centroids = kmeans.cluster_centers_
# Create dataframes for different clusters
df_cluster1 = principal_df[y_kmeans == 0]
df_cluster2 = principal_df[y_kmeans == 1]
df_cluster3 = principal_df[y_kmeans == 2]
# Plot the clusters and centroids
plt.scatter(df_cluster1['PC2'], df_cluster1['PC3'], s=50, c='blue', label='Cluster 1')
plt.scatter(df_cluster2['PC2'], df_cluster2['PC3'], s=50, c='green', label='Cluster 2')
plt.scatter(df_cluster3['PC2'], df_cluster3['PC3'], s=50, c='red', label='Cluster 3')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black', label='Centroids')
plt.xlabel('PC2')
plt.ylabel('PC3')
plt.title('K-means Clustering of Wine Dataset (PC2 vs PC3)')
plt.legend()
plt.show()

#calculate silhoutte score average
from sklearn.metrics import silhouette_score
sil_score=silhouette_score(principal_df, y_kmeans)
print(sil_score)

from sklearn.metrics import davies_bouldin_score

# Calculate Davies-Bouldin Index
db_index = davies_bouldin_score(df_scaled, y_kmeans)
print(f'Davies-Bouldin Index: {db_index}')