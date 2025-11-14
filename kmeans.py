import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Read in the processed data / Print head
df = pd.read_csv("processed_user_item_matrix.csv")

# Data preprocessing for kmeans
user_ids = df['user_id']
X = df.drop(columns= ['user_id'])

# Standardization
std_scaler = StandardScaler()
X_scaled = std_scaler.fit_transform(X)

# Dimensionality reduction
pca = PCA(n_components=0.9)
X_pca = pca.fit_transform(X_scaled)

# Create a final DataFrame and save it
pca_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_final = pd.DataFrame(X_pca, columns=pca_columns)
X_final.insert(0, 'user_id', user_ids)


print(X_final.head())
X_features = X_final.drop(columns=['user_id'])

# Visualize data
fig = plt.figure(0)
plt.grid(True)
plt.scatter(X_features.iloc[:,0],X_features.iloc[:,1])
plt.show()
