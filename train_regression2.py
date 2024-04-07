import json
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, [2, 3]]  # Petal length and sepal width
y = iris.data[:, 3]  # Petal width

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from LinearRegression import LinearRegression

# Create a Linear Regression model
linear_model = LinearRegression()

# Train the model using the training set
weights, bias, loss, step_numbers = linear_model.fit(X_train, y_train)

params = {
            'weights': weights.tolist(),
            'bias': bias
        }
with open('model_params2.json', 'w') as f:
    json.dump(params, f)

# Plot the loss values against the step numbers
plt.plot(step_numbers, loss)
plt.xlabel('Step Number')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()