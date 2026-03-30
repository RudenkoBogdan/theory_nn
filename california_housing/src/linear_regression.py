import numpy as np

class LinearRegression:
    def __init__(self,
                 learning_rate=1e-3,
                 n_iter = 2000,
                 metric = 'mse',
                 random_seed = 42):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.metric = metric
        self.weights = None 
        self.bias = None
        self.height = None
        self.width = None
        self.loss_history = None
        self.rng = np.random.default_rng(seed=random_seed)
    
    def mse(self, y_pred, y):
        return np.mean((y - y_pred)**2)
    
    def mae(self, y_pred, y):
        return np.mean(np.abs(y - y_pred))
    
    def rmse(self, y_pred, y):
        return np.sqrt(self.mse(y_pred, y))
    
    def grad(self, X, y_pred, y):
        metric = self.metric
        delta = y_pred - y
        if metric == 'mse':
            return (2 / self.height) * X.T @ delta
        elif metric == 'mae':
            return (1 / self.height) * X.T @ np.sign(delta)
        else:
            mse = np.mean(delta**2)
            return (1 / (2 * mse)) * ( (2 / self.height) * X.T @ delta)
    
    def loss_function(self, y_pred, y):
        metric = self.metric
        if metric == 'mse':
            return self.mse(y_pred, y)
        elif metric == 'mae':
            return self.mae(y_pred, y)
        else:
            return self.rmse(y_pred, y)
    
    def predict(self, X):
        return X @ self.weights + self.bias

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y).flatten()
        height, width = X.shape
        self.height, self.width = height, width
        self.weights = self.rng.standard_normal(width) * 0.01
        self.bias = 0.0

        loss_history = []
        for _ in range(self.n_iter):
            y_pred = self.predict(X)
            loss_history.append(self.loss_function(y_pred, y))
            self.weights -= self.learning_rate * self.grad(X, y_pred, y)
            self.bias -= self.learning_rate * (2 / self.height) * np.sum(y_pred - y)
        return loss_history

    def get_weights(self):
        return self.weights, self.bias