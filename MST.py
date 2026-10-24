
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from mst_clustering import MSTClustering

warnings.filterwarnings("ignore", message="elementwise")

# Create dataset using a DataFrame
df = pd.DataFrame({
    "Feature1": [
        1.0, 1.5, 2.0, 2.5, 3.0,
        8.0, 8.5, 9.0, 9.5, 10.0,
        4.0, 4.5, 5.0, 5.5, 6.0,
        12.0, 12.5, 13.0, 13.5, 14.0
    ],
    "Feature2": [
        1.0, 1.8, 1.5, 2.2, 1.0,
        8.0, 8.5, 9.0, 8.2, 9.5,
        12.0, 12.5, 13.0, 12.2, 13.5,
        3.0, 3.5, 4.0, 3.2, 4.5
    ]
})

print("Dataset:")
print(df.head())
print("\nDataset Shape:", df.shape)

# Select features
X = df[["Feature1", "Feature2"]].to_numpy()


# Function to plot the Minimum Spanning Tree
def plot_mst(model, cmap="rainbow"):
    # Correct attribute name
    X = model.X_fit_

    fig, ax = plt.subplots(
        1, 2, figsize=(14, 6), sharex=True, sharey=True
    )

    for axi, full_graph, colors in zip(
        ax, [True, False], ["lightblue", model.labels_]
    ):
        segments = model.get_graph_segments(
            full_graph=full_graph
        )

        # Draw MST edges
        axi.plot(
            segments[0], segments[1],
            "-k", zorder=1, linewidth=1
        )

        # Draw data points
        if full_graph:
            axi.scatter(
                X[:, 0], X[:, 1],
                c=colors, zorder=2,
                edgecolors="black"
            )
        else:
            axi.scatter(
                X[:, 0], X[:, 1],
                c=colors, cmap=cmap,
                zorder=2, edgecolors="black"
            )

        axi.set_xlabel("Feature 1")
        axi.set_ylabel("Feature 2")
        axi.axis("tight")
        axi.grid(True, linestyle="--", alpha=0.3)

    ax[0].set_title("Full Minimum Spanning Tree", size=14)
    ax[1].set_title("Minimum Spanning Tree (Cut)", size=14)

    plt.tight_layout()


# Initialize the clustering model
model = MSTClustering(
    cutoff_scale=2,
    approximate=False
)

# Fit model and obtain cluster labels
model.fit(X)
labels = model.labels_

# Add cluster labels to DataFrame
df["Cluster"] = labels

print("\nDataFrame with Cluster Labels:")
print(df)

# Plot both MST visualizations
plot_mst(model)

# Display plots
plt.show()
