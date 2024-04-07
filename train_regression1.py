import json
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, [0, 2]]  # Sepal width and petal length
y = iris.data[:, 0]  # Target variable: sepal length

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from LinearRegression import LinearRegression

# Create a Linear Regression model
linear_model = LinearRegression()

# Train the model using the training set
weights_no_reg, bias_no_reg, loss, step_numbers = linear_model.fit(X_train, y_train)

linear_model_with_reg = LinearRegression(regularization=0.1)  # Adjust the regularization parameter as needed
weights_with_reg, bias_with_reg, _, _ = linear_model_with_reg.fit(X_train, y_train)

params = {
            'weights': weights_no_reg.tolist(),
            'bias': bias_no_reg
        }
with open('model_params1.json', 'w') as f:
    json.dump(params, f)

parameter_difference = weights_no_reg - weights_with_reg
bias_difference = bias_no_reg - bias_with_reg
print("Difference in parameters without regularization and with regularization:")
print("Weights Difference:", parameter_difference)
print("Bias Difference:", bias_difference)

# Plot the loss values against the step numbers
plt.plot(step_numbers, loss)
plt.xlabel('Step Number')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()


