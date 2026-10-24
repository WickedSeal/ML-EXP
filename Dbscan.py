
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

# Create a DataFrame with four clearly separated clusters
df = pd.DataFrame({
    "CustomerID": range(1, 25),
    "Annual Income (k$)": [
        15, 16, 17, 18, 19, 20,       # Cluster 1
        35, 36, 37, 38, 39, 40,       # Cluster 2
        60, 61, 62, 63, 64, 65,       # Cluster 3
        85, 86, 87, 88, 89, 90        # Cluster 4
    ],
    "Spending Score (1-100)": [
        15, 16, 17, 18, 19, 20,       # Cluster 1
        75, 76, 77, 78, 79, 80,       # Cluster 2
        15, 16, 17, 18, 19, 20,       # Cluster 3
        75, 76, 77, 78, 79, 80        # Cluster 4
    ]
})

print(df.head())
print("Dataset Shape:", df.shape)

# Select features
x = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

print("Feature Shape:", x.shape)

# Calculate nearest-neighbor distances
neighb = NearestNeighbors(n_neighbors=2)
nbrs = neighb.fit(x)
distances, indices = nbrs.kneighbors(x)

# Sort second-nearest-neighbor distances
distances = np.sort(distances[:, 1])

# Plot k-distance graph
plt.figure(figsize=(7, 4))
plt.plot(distances, marker="o")
plt.title("K-Distance Graph")
plt.xlabel("Data Points (sorted)")
plt.ylabel("Distance to 2nd Nearest Neighbor")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Apply DBSCAN
dbscan = DBSCAN(eps=3, min_samples=3)
labels = dbscan.fit_predict(x)

# Plot clusters
plt.figure(figsize=(8, 5))

scatter = plt.scatter(
    x[:, 0],
    x[:, 1],
    c=labels,
    cmap="plasma",
    s=100,
    edgecolors="black"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("DBSCAN Clustering — Four Separated Clusters")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# Display cluster labels
df["Cluster"] = labels

print("\nDataFrame with Cluster Labels:")
print(df)

print(
    "\nNumber of clusters:",
    len(set(labels)) - (1 if -1 in labels else 0)
)
print("Number of noise points:", list(labels).count(-1))
