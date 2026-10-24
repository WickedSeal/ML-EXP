from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 1. Load dataset & split
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train AdaBoost Classifier
ada_clf = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
ada_clf.fit(X_train, y_train)

# --- VISUALIZATION 1: Feature Importances ---
# Shows which columns/features contributed the most to the boosting process
plt.figure(figsize=(8, 5))
plt.barh(data.feature_names, ada_clf.feature_importances_)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("AdaBoost Feature Importances")
plt.show()

# --- VISUALIZATION 2: Plotting one of the weak learners (e.g., the 1st tree) ---
# AdaBoost stores all its weak trees in the .estimators_ list
plt.figure(figsize=(12, 8))
plot_tree(
    ada_clf.estimators_[0], 
    feature_names=data.feature_names, 
    class_names=data.target_names, 
    filled=True
)
plt.title("Visualization of the 1st Weak Learner (Decision Stump) in AdaBoost")
plt.show()