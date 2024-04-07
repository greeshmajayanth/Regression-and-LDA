import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, [0, 2]]  # Sepal width and petal length
y = iris.data[:, 0]  # Target variable: sepal length

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the model parameters
with open('model_params1.json', 'r') as f:
    params = json.load(f)
weights = np.array(params['weights'])
bias = np.array(params['bias'])

# Predict on the test set
y_pred = np.dot(X_test, weights) + bias

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Print the mean squared error
print('Mean Squared Error for regression 1:', mse)
