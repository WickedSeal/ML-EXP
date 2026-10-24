from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
X = np.array([[1], [2], [3], [4], [5], [6]])
y = [0, 0, 0, 1, 1, 1]
model = LogisticRegression()
model.fit(X, y)
X_curve = np.linspace(1, 6, 200).reshape(-1, 1)
Y_curve = model.predict_proba(X_curve)[:, 1]
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_curve, Y_curve, color='red', label='S-Curve')
plt.xlabel('X')
plt.ylabel('Probability')
plt.title('Logistic Regression')
plt.legend()
plt.show()
