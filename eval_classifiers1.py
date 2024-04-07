import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score

from LogisticRegression import LogisticRegression
from LDA import LinearDiscriminantAnalysis

# Load the Iris dataset
iris = load_iris()

# Extract the petal length and petal width features
X = iris.data[:, 2:4]  # Petal length and petal width
y = iris.target

# Split the dataset into training and test sets, ensuring an even split of each class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Logistic Regression model and fit it to the training data
model_lr = LogisticRegression()

model_lr.fit(X_train, y_train)
pred2 = model_lr.predict(X_test)

# Compute the accuracy of Logistic Regression on the test set
lr_accuracy = (accuracy_score(y_test, pred2))*100

# Create the LDA model and fit it to the training data
model_lda = LinearDiscriminantAnalysis()

model_lda.fit(X_train, y_train)

y_pred = model_lda.predict(X_test)
# Compute the accuracy of LDA on the test set
lda_accuracy = (accuracy_score(y_test, y_pred))*100

# Print the accuracies
print("Logistic Regression Accuracy (Petal Length/Width):" + str(lr_accuracy) + '%')
print("LDA Accuracy (Petal Length/Width):" + str(lda_accuracy) + '%')

# Visualize decision regions for Logistic Regression using petal length/width
plot_decision_regions(X_train, y_train, clf=model_lr)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Logistic Regression - Petal Length/Width')
plt.show()

# Visualize decision regions for Linear Discriminant Analysis using petal length/width
plot_decision_regions(X_train, y_train, clf=model_lda)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Linear Discriminant Analysis - Petal Length/Width')
plt.show()
