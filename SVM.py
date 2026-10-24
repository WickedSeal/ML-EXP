
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# 1. Create Dataset Using DataFrame
# -------------------------------

data = pd.DataFrame({
    "Feature1": [
        1.0, 1.5, 2.0, 2.5, 3.0,
        3.5, 4.0, 4.5, 5.0, 5.5,
        6.0, 6.5, 7.0, 7.5, 8.0
    ],
    "Feature2": [
        2.0, 2.5, 1.5, 3.0, 2.0,
        3.5, 3.0, 4.0, 3.5, 5.0,
        4.5, 5.5, 5.0, 6.0, 5.5
    ],
    "Class": [
        "A", "A", "A", "A", "A",
        "A", "B", "B", "B", "B",
        "B", "B", "B", "B", "B"
    ]
})

print("Dataset:")
print(data)

# -------------------------------
# 2. Select Features and Target
# -------------------------------

print("\nAvailable columns:", list(data.columns))

target = input("Enter target column name: ")

if target not in data.columns:
    raise ValueError("Target column not found!")

# Select two numeric features for 2D visualization
features = data.drop(columns=[target]).select_dtypes(
    include=np.number
).columns.tolist()

if len(features) < 2:
    raise ValueError(
        "At least two numerical feature columns are required."
    )

print("\nAvailable numerical features:", features)

feature1 = input(
    f"Enter first feature ({features[0]}): "
).strip() or features[0]

feature2 = input(
    f"Enter second feature ({features[1]}): "
).strip() or features[1]

if feature1 == feature2:
    raise ValueError("Select two different features.")

if feature1 not in features or feature2 not in features:
    raise ValueError("Invalid feature selected.")

# Keep selected features and target; remove missing rows
data = data[[feature1, feature2, target]].dropna().copy()

X = data[[feature1, feature2]].to_numpy(dtype=float)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data[target])

if len(np.unique(y)) < 2:
    raise ValueError("The target must contain at least two classes.")

# -------------------------------
# 3. Split Dataset
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# 4. Train SVM Model
# -------------------------------

svm_clf = SVC(kernel="linear", C=1.0)
svm_clf.fit(X_train, y_train)

print("\nTraining completed successfully.")
print("Number of support vectors:", len(svm_clf.support_vectors_))
print("Support vectors per class:", svm_clf.n_support_)

# -------------------------------
# 5. Create Grid for Visualization
# -------------------------------

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# Predict the class for each grid point
Z = svm_clf.predict(grid)
Z = Z.reshape(xx.shape)

# -------------------------------
# 6. Visualize Decision Boundary
# -------------------------------

plt.figure(figsize=(10, 7))

colors = plt.cm.coolwarm(
    np.linspace(0.1, 0.9, len(label_encoder.classes_))
)

# Color the decision regions
plt.contourf(
    xx, yy, Z,
    alpha=0.3,
    levels=np.arange(len(label_encoder.classes_) + 1) - 0.5,
    cmap=plt.cm.coolwarm
)

# Plot data points for each class
for class_id, class_name in enumerate(label_encoder.classes_):
    mask = y == class_id

    plt.scatter(
        X[mask, 0],
        X[mask, 1],
        color=colors[class_id],
        edgecolors="black",
        s=60,
        label=str(class_name),
        zorder=3
    )

# Highlight support vectors
plt.scatter(
    svm_clf.support_vectors_[:, 0],
    svm_clf.support_vectors_[:, 1],
    s=200,
    facecolors="none",
    edgecolors="black",
    linewidth=2,
    label="Support Vectors",
    zorder=4
)

# Draw the linear decision boundary or boundaries
if len(label_encoder.classes_) == 2:
    # Binary SVM: decision function equals zero at the hyperplane
    decision = svm_clf.decision_function(grid)
    decision = decision.reshape(xx.shape)

    plt.contour(
        xx, yy, decision,
        levels=[0],
        colors="black",
        linewidths=2
    )
else:
    # Multiclass SVM: draw pairwise decision boundaries
    for i in range(len(label_encoder.classes_)):
        for j in range(i + 1, len(label_encoder.classes_)):
            pair_mask = np.isin(y, [i, j])

            pair_model = SVC(kernel="linear", C=1.0)
            pair_model.fit(X[pair_mask], y[pair_mask])

            pair_decision = pair_model.decision_function(grid)
            pair_decision = pair_decision.reshape(xx.shape)

            plt.contour(
                xx, yy, pair_decision,
                levels=[0],
                colors="black",
                linewidths=1.5,
                linestyles="solid"
            )

plt.title("SVM Hyperplane and Support Vectors", fontsize=15)
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# 7. Predict a New Sample
# -------------------------------

print("\nEnter values for a new sample:")

value1 = float(input(f"{feature1}: "))
value2 = float(input(f"{feature2}: "))

new_sample = np.array([[value1, value2]])

prediction = svm_clf.predict(new_sample)[0]

print(
    "\nPredicted class:",
    label_encoder.inverse_transform([prediction])[0]
)





# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC

# # 1. Load data and keep only 2 features & 2 classes for a clear 2D line hyperplane
# iris = load_iris()
# X = iris.data[:100, :2]  # Taking first 100 samples (Class 0 and Class 1) and 2 features
# y = iris.target[:100]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 2. Train a Linear SVM model
# svm_clf = SVC(kernel='linear', C=1.0)
# svm_clf.fit(X_train, y_train)

# # 3. Plotting the Hyperplane and Support Vectors
# plt.figure(figsize=(8, 6))

# # Create a grid of points to evaluate the decision boundary
# x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
# y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
#                      np.arange(y_min, y_max, 0.02))

# # Predict cluster/class for every point on the grid
# Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# # Draw the Hyperplane region (Decision Boundary background)
# plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# # Plot all data points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=50)

# # HIGHLIGHT THE SUPPORT VECTORS (The points that define the hyperplane)
# plt.scatter(svm_clf.support_vectors_[:, 0], svm_clf.support_vectors_[:, 1], 
#             s=200, facecolors='none', edgecolors='black', linewidth=2, label='Support Vectors')

# plt.title("SVM Hyperplane & Support Vectors Visualization")
# plt.xlabel("Feature 1 (Sepal Length)")
# plt.ylabel("Feature 2 (Sepal Width)")
# plt.legend()
# plt.show()