import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import SpectralClustering

# 1. Load dataset (unsupervised: we ignore target labels during clustering)
data = load_iris()
X = data.data

# 2. Initialize and train Graph-Based Spectral Clustering
# affinity='rbf' constructs a similarity graph using the Radial Basis Function kernel
spectral_clf = SpectralClustering(n_clusters=3, affinity='rbf', random_state=42)
cluster_labels = spectral_clf.fit_predict(X)

# 3. Print assigned cluster labels for the first 20 samples
print("Assigned Cluster Labels (First 20 samples):")
print(cluster_labels[:20])

# 4. Visualization of the Graph-Based Clustering Results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='rainbow', edgecolor='k', s=70)
plt.title("Graph-Based Spectral Clustering (Iris Dataset)")
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()