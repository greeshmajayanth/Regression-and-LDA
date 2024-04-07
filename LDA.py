import numpy as np

class LinearDiscriminantAnalysis:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = X.shape
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)

        class_means = []
        class_covs = []
        class_priors = []

        for class_label in unique_classes:
            X_class = X[y == class_label]
            class_means.append(np.mean(X_class, axis=0))
            class_covs.append(np.cov(X_class.T))
            class_priors.append(len(X_class) / m)

        total_mean = np.mean(X, axis=0)
        within_class_cov = sum(class_covs)
        between_class_cov = sum(
            [class_priors[i] * np.outer(class_means[i] - total_mean, class_means[i] - total_mean)
             for i in range(num_classes)]
        )

        # Inverse of within-class covariance matrix
        inv_within_class_cov = np.linalg.inv(within_class_cov)

        # Solving for weights and bias
        self.weights = np.dot(inv_within_class_cov, (class_means[1] - class_means[0]))
        self.bias = -0.5 * np.dot(self.weights.T, (class_means[0] + class_means[1]))

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = np.where(linear_output >= 0, 1, 0)
        return y_pred

    
    def accuracy(self, X_test, Y_test):
        predicted_labels = self.predict(X_test)
        correct_predictions = np.sum(predicted_labels == Y_test)
        total_predictions = len(Y_test)
        accuracy = correct_predictions / total_predictions
        return accuracy*100
