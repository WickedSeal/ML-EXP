import pandas as pd
from sklearn import datasets 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
# Load the Iris dataset 
iris = datasets.load_iris()
target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
# Create a DataFrame from the Iris dataset
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_names'] = df['target'].map(target_names)
# Countplot of target names
sns.countplot(x='target_names', data=df)
plt.title('Iris Target Value Count') 
plt.show()
# Standardize the feature data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df[iris.feature_names])
# Apply PCA with 3 components
pca = PCA(n_components=3)
pca_features = pca.fit_transform(x_scaled)
# Create a DataFrame from the PCA features
pca_df = pd.DataFrame(data=pca_features, columns=['PC1', 'PC2', 'PC3'])
pca_df['target'] = iris.target
pca_df['target'] = pca_df['target'].map(target_names)
plt.bar(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_)
plt.xlabel('PCA Feature')
plt.ylabel('Explained Variance')
plt.title('Feature Explained Variance')
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.decomposition import PCA

# # 1. Load dataset (4 features: sepal length/width, petal length/width)
# iris = load_iris()
# X = iris.data
# y = iris.target

# print(f"Original Data Shape (Features): {X.shape}")

# # 2. Initialize and apply PCA to reduce 4 dimensions down to 2
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# print(f"Reduced Data Shape (Components): {X_pca.shape}")
# print(f"Explained Variance Ratio (Information retained): {pca.explained_variance_ratio_}")

# # 3. Visualization of PCA projection
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=70)
# plt.title("Principal Component Analysis (PCA) on Iris Dataset")
# plt.xlabel("Principal Component 1 (PC1)")
# plt.ylabel("Principal Component 2 (PC2)")
# plt.colorbar(scatter, label='Actual Iris Classes')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()