from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
X = [[1], [2], [3], [4], [5]]
y = [12, 18, 25, 35, 42]
model = LinearRegression()
model.fit(X, y)
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.show()
print("Prediction:", model.predict([[6]]))
