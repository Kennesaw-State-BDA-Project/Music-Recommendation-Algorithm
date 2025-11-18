import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Read in the processed data / Print head
init_df = pd.read_csv("processed_user_item_matrix.csv")



# Data preprocessing for kmeans
user_ids = init_df['user_id']
X = init_df.drop(columns= ['user_id'])
print(f"Original feature matrix shape: {X.shape}")

"""
fig = plt.figure(0)
plt.grid(True)
plt.scatter(X.iloc[:,0],X.iloc[:,1])
plt.show()
"""

# - Log transformation
X_log = X.apply(np.log1p)

# - Standardization
std_scaler = StandardScaler()
X_scaled = std_scaler.fit_transform(X_log)

# - Dimensionality reduction
pca = PCA(n_components=0.9)
X_pca = pca.fit_transform(X_scaled)

# - Create a final DataFrame and save it
pca_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
df = pd.DataFrame(X_pca, columns=pca_columns)
df.insert(0, 'user_id', user_ids)

# Print the final processed dataframe
X_features = df.drop(columns=['user_id'])
print(X_features.shape)
print(X_features)

fig = plt.figure(0)
plt.grid(True)
plt.scatter(X_features.iloc[:,0],X_features.iloc[:,1])
plt.show()

# Determine the optimal number of clusters, k, (Elbow Method)
inertia = []
k_range = range(1,11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=44, n_init=10)
    kmeans.fit(X_features)
    inertia.append(kmeans.inertia_)

plt.plot(k_range, inertia, marker= 'o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Run k-means clustering algorithm

k = 3
kmeans = KMeans(n_clusters= k, random_state=0, n_init=10)
kmeans.fit(X_features)

# Analyze and visualize the results

# Add cluster labels to df
df['cluster'] = kmeans.labels_

# Print cluster counts
print("Cluster counts:")
print(df['cluster'].value_counts())

# 2D scatter plot of clusters using first two PCA components
plt.figure(figsize=(8, 6))
plt.grid(True)

scatter = plt.scatter(
    df['PC1'],
    df['PC2'],
    c=df['cluster'],
    cmap='viridis',
    alpha=0.7
)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('K-Means Clustering')
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()