import numpy as np

class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        # Initialize attributes
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Shape of X: (number of training examples: m, number of features: n)
        m, n = X.shape
        
        # Initializing weights as a matrix of zeros of size: (number of features: n, 1) and bias as 0
        self.weights = np.zeros((n, 1))
        self.bias = 0
        
        # Reshaping y as (m, 1) in case your dataset is initialized as (m,) which can cause problems
        y = y.reshape(m, 1)
        
        # Set aside 10% of the training data as the validation set
        val_size = int(0.1 * m)
        X_train, X_val = X[val_size:], X[:val_size]
        y_train, y_val = y[val_size:], y[:val_size]

        # Calculate the number of batches
        num_batches = len(X_train) // self.batch_size
        
        # Initializing variables for early stopping
        best_loss = float('inf')
        consecutive_increases = 0
        
        # Empty list to store losses so we can plot them later against epochs
        losses = []
        step_numbers = []
        
        # Gradient Descent loop / Training loop
        for epoch in range(self.max_epochs):
            # Calculate the indices for shuffling the training data
            shuffled_indices = np.random.permutation(len(X))
            X_train_shuffled = X[shuffled_indices]
            y_train_shuffled = y[shuffled_indices]
            
            # Mini-batch Gradient Descent
            for i in range(0, m - val_size, self.batch_size):
                X_batch = X_train_shuffled[i:i+self.batch_size]
                y_batch = y_train_shuffled[i:i+self.batch_size]
                
                # Calculating prediction: y_hat or h(x)
                y_hat = np.dot(X_batch, self.weights) + self.bias
                
                # Calculating loss with L2 regularization
                loss = np.mean((y_hat - y_batch) ** 2) + (self.regularization / (2 * m)) * np.sum(self.weights ** 2)
                losses.append(loss)
                step_numbers.append(epoch)
                
                # Calculating derivatives of parameters (weights and bias) with L2 regularization
                dw = (1/m) * np.dot(X_batch.T, (y_hat - y_batch)) + (self.regularization / m) * self.weights
                db = (1/m) * np.sum(y_hat - y_batch)
                
                # Updating the parameters: parameter := parameter - lr * derivative of loss/cost w.r.t parameter
                self.weights -= 0.01 * dw
                self.bias -= 0.01 * db
            
            # Calculating loss on the validation set
            y_val_hat = np.dot(X_val, self.weights) + self.bias
            val_loss = np.mean((y_val_hat - y_val) ** 2) + (self.regularization / (2 * m)) * np.sum(self.weights ** 2)
            
            # Check for early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                consecutive_increases = 0
                # Save the current model parameters
                best_weights = self.weights
                best_bias = self.bias
            else:
                consecutive_increases += 1
                if consecutive_increases == self.patience:
                    # Stop training if the validation loss increases for `patience` consecutive epochs
                    break
        
        # Set the model parameters to the best parameters obtained during training
        self.weights = best_weights
        self.bias = best_bias
        
        # Returning the parameters and losses for further analysis
        return self.weights, self.bias, losses, step_numbers

    def predict(self, X):
        # Calculate the predicted values based on the learned weights and bias
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        # Evaluate the linear model using the mean squared error
        predictions = self.predict(X)
        n = len(y)
        
        if len(y.shape) == 1:  # 1D array
            mse = np.mean((y - predictions) ** 2)
        else:  # 2D array
            m = y.shape[1]
            mse = np.sum((y - predictions) ** 2) / (n * m)

        return mse
