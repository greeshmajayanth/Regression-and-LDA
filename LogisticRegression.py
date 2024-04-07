import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.max_iter):
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_output)
            
            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_output)
        y_pred = np.round(y_pred)
        return y_pred.astype(int)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def accuracy(self, X, y):
        # Apply the predict method to get the predicted labels
        y_pred = self.predict(X)
        
        # Calculate the accuracy by comparing the predicted labels with the true labels
        accuracy = np.mean(y_pred == y)
        
        return accuracy*100