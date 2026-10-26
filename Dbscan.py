
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
data = {
    "Annual Income (k$)": [
        15, 16, 17, 18, 19, 20,
        35, 36, 37, 38, 39, 40,
        60, 61, 62, 63, 64, 65,
        85, 86, 87, 88, 89, 90
    ],
    "Spending Score (1-100)": [
        15, 16, 17, 18, 19, 20,
        75, 76, 77, 78, 79, 80,
        15, 16, 17, 18, 19, 20,
        75, 76, 77, 78, 79, 80
    ]
}
df = pd.DataFrame(data)
X = StandardScaler().fit_transform(df)
db = DBSCAN(eps=0.3, min_samples=3)
df["Cluster"] = db.fit_predict(X)
print(df)
print("Clusters:", set(df["Cluster"]) - {-1})
plt.scatter(df.iloc[:, 0], df.iloc[:, 1],c=df["Cluster"], cmap="viridis")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("DBSCAN Clustering")
plt.show()