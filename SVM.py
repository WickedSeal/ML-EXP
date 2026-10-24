import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 1. Load data and keep only 2 features & 2 classes for a clear 2D line hyperplane
iris = load_iris()
X = iris.data[:100, :2]  # Taking first 100 samples (Class 0 and Class 1) and 2 features
y = iris.target[:100]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train a Linear SVM model
svm_clf = SVC(kernel='linear', C=1.0)
svm_clf.fit(X_train, y_train)

# 3. Plotting the Hyperplane and Support Vectors
plt.figure(figsize=(8, 6))

# Create a grid of points to evaluate the decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict cluster/class for every point on the grid
Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Draw the Hyperplane region (Decision Boundary background)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# Plot all data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=50)

# HIGHLIGHT THE SUPPORT VECTORS (The points that define the hyperplane)
plt.scatter(svm_clf.support_vectors_[:, 0], svm_clf.support_vectors_[:, 1], 
            s=200, facecolors='none', edgecolors='black', linewidth=2, label='Support Vectors')

plt.title("SVM Hyperplane & Support Vectors Visualization")
plt.xlabel("Feature 1 (Sepal Length)")
plt.ylabel("Feature 2 (Sepal Width)")
plt.legend()
plt.show()