# =====================================================================
# ALL PRACTICALS IMPLEMENTATION WITH VISUALIZATIONS
# =====================================================================

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Common Datasets
iris = load_iris()
X_c, y_c = iris.data, iris.target
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

diabetes = load_diabetes()
X_r, y_r = diabetes.data, diabetes.target
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_r, y_r, test_size=0.2, random_state=42)


# ---------------------------------------------------------------------
# 2. Linear Regression + Visualization (Scatter Plot & Best Fit Line)
# ---------------------------------------------------------------------
print("--- 2. Linear Regression ---")
from sklearn.linear_model import LinearRegression

# Use a single feature (e.g., body mass index) for easy 2D line plotting
X_lin_train = X_tr_r[:, np.newaxis, 2]
X_lin_test = X_te_r[:, np.newaxis, 2]

lin_reg = LinearRegression()
lin_reg.fit(X_lin_train, y_tr_r)
y_pred_r = lin_reg.predict(X_lin_test)

# Visualization
plt.figure(figsize=(7, 4))
plt.scatter(X_lin_test, y_te_r, color='black', label='Actual Data')
plt.plot(X_lin_test, y_pred_r, color='blue', linewidth=3, label='Regression Line')
plt.title("2. Linear Regression Fit")
plt.xlabel("Feature Value")
plt.ylabel("Target Value")
plt.legend()
plt.show()


# ---------------------------------------------------------------------
# 3. Logistic Regression + Visualization (Decision Boundary / Scatter)
# ---------------------------------------------------------------------
print("--- 3. Logistic Regression ---")
from sklearn.linear_model import LogisticRegression

# Use first 2 features of Iris for 2D visualization
X_log = X_c[:, :2]
X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(X_log, y_c, test_size=0.2, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_tr_l, y_tr_l)

# Visualization: Scatter plot of classes
plt.figure(figsize=(7, 4))
plt.scatter(X_te_l[:, 0], X_te_l[:, 1], c=y_te_l, cmap='viridis', edgecolor='k')
plt.title("3. Logistic Regression Classification (First 2 Iris Features)")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()


# ---------------------------------------------------------------------
# 4. Decision Tree + Visualization (Tree Structure Plot)
# ---------------------------------------------------------------------
print("--- 4. Decision Tree ---")
from sklearn.tree import DecisionTreeClassifier, plot_tree

dt_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_clf.fit(X_tr_c, y_tr_c)

# Visualization: Decision Tree Diagram
plt.figure(figsize=(10, 6))
plot_tree(dt_clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("4. Decision Tree Structure Visualization")
plt.show()


# ---------------------------------------------------------------------
# 5. Ensemble Learning (AdaBoost) + Visualization (Feature Importances)
# ---------------------------------------------------------------------
print("--- 5. Ensemble Learning: AdaBoost ---")
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_clf.fit(X_tr_c, y_tr_c)

# Visualization: Feature Importances Bar Chart
plt.figure(figsize=(7, 4))
plt.barh(iris.feature_names, ada_clf.feature_importances_, color='teal')
plt.title("5. AdaBoost Feature Importances")
plt.xlabel("Importance Score")
plt.show()


# ---------------------------------------------------------------------
# 6. Support Vector Machine (SVM) + Visualization (Decision Scatter)
# ---------------------------------------------------------------------
print("--- 6. Support Vector Machine (SVM) ---")
from sklearn.svm import SVC

svm_clf = SVC(kernel='rbf', random_state=42)
svm_clf.fit(X_tr_l, y_tr_l)

# Visualization: SVM predictions on 2D space
plt.figure(figsize=(7, 4))
plt.scatter(X_te_l[:, 0], X_te_l[:, 1], c=y_te_l, cmap='coolwarm', edgecolor='k')
plt.title("6. Support Vector Machine (SVM) Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# ---------------------------------------------------------------------
# 7. Graph-Based Clustering (Spectral Clustering) + Visualization
# ---------------------------------------------------------------------
print("--- 7. Graph-Based Clustering (Spectral Clustering) ---")
from sklearn.cluster import SpectralClustering

# Unsupervised graph-based clustering
spectral = SpectralClustering(n_clusters=3, affinity='rbf', random_state=42)
cluster_labels = spectral.fit_predict(X_c)

# Visualization: Scatter plot colored by generated clusters
plt.figure(figsize=(7, 4))
plt.scatter(X_c[:, 0], X_c[:, 1], c=cluster_labels, cmap='rainbow', edgecolor='k')
plt.title("7. Graph-Based Spectral Clustering Output")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()