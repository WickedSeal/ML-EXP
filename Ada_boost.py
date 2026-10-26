from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
ada_clf = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
ada_clf.fit(X_train, y_train)
plt.figure(figsize=(8, 5))
plt.barh(data.feature_names, ada_clf.feature_importances_)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("AdaBoost Feature Importances")
plt.show()
# y_pred = ada_clf.predict(X_test)
# print("Predictions:", y_pred)
# print("Actual:", y_test)