from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from LogisticRegression import LogisticRegression
from LDA import LinearDiscriminantAnalysis

# Load the Iris dataset
iris = load_iris()

# Split the dataset into features (X) and target (y)
X = iris.data
y = iris.target

# Randomly split the data into training and test sets, ensuring an even split of each class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Logistic Regression model and fit it to the training data
modellr = LogisticRegression()

modellr.fit(X_train, y_train)
pred2 = modellr.predict(X_test)

# Compute the accuracy of Logistic Regression on the test set
lr_accuracy = (accuracy_score(y_test, pred2))*100

# Create the LDA model and fit it to the training data
modellda = LinearDiscriminantAnalysis()

modellda.fit(X_train, y_train)

y_pred = modellda.predict(X_test)

# Compute the accuracy of LDA on the test set
lda_accuracy = (accuracy_score(y_test, y_pred))*100

# Print the accuracies
print("Logistic Regression Accuracy for all features:" + str(lr_accuracy) + '%')
print("LDA Accuracy for all features:" + str(lda_accuracy) + '%')

