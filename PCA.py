import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# 1. Load dataset (4 features: sepal length/width, petal length/width)
iris = load_iris()
X = iris.data
y = iris.target

print(f"Original Data Shape (Features): {X.shape}")

# 2. Initialize and apply PCA to reduce 4 dimensions down to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"Reduced Data Shape (Components): {X_pca.shape}")
print(f"Explained Variance Ratio (Information retained): {pca.explained_variance_ratio_}")

# 3. Visualization of PCA projection
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=70)
plt.title("Principal Component Analysis (PCA) on Iris Dataset")
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.colorbar(scatter, label='Actual Iris Classes')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()