import numpy as np

class Regressor:
    def __init__(self,
                 learning_rate=1e-3,
                 epochs = 500,
                 metric = 'mse',
                 depth = 4,
                 width = 32,
                 activation_func = 'relu',
                 loss_func = 'mse',
                 weight_decay = 1e-5,
                 random_seed = 42):
        
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.metric = metric
        self.depth = depth
        self.width = width
        self.activation_func = activation_func
        self.loss_func = loss_func
        self.weight_decay = weight_decay

        self.weights = None 
        self.bias = None 
        self.activation = None
        self.loss = None
        self.grad = None
        self.cache = None

        self.rng = np.random.default_rng(random_seed)

    def _init_weights(self):
        self.weights = []
        self.bias = []
        width = self.width
        self.weights.append(self.rng.normal(size=(8, width)))
        self.bias.append(self.rng.normal(size=(1, width)))
        for _ in range(self.depth - 2):
            self.weights.append(self.rng.normal(size=(width, width)))
            self.bias.append(self.rng.normal(size=(1, width)))
        self.weights.append(self.rng.normal(size=(width, 1)))
        self.bias.append(self.rng.normal(size=(1, 1)))
        return self
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_grad(self, x):
        result = np.zeros_like(x)
        result[x > 0] = 1
        return result

    def leaky_relu(self, x, a=0.1):
        return np.maximum(x, a * x)
    
    def leaky_relu_grad(self, x, a=0.1):
        result = np.ones_like(x)
        result[x <= 0] = a
        return result

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_grad(self, x):
        sigma = self.sigmoid(x)
        return sigma * (1 - sigma)
    
    def _set_activation(self):
        name = self.activation_func
        if name == 'relu':
            self.activation = self.relu
            self.activation_grad = self.relu_grad
        elif name == 'leaky_relu':
            self.activation = self.leaky_relu
            self.activation_grad = self.leaky_relu_grad
        else:
            self.activation = self.sigmoid
            self.activation_grad = self.sigmoid_grad

    def mse(self, y, y_pred):
        return np.mean((y - y_pred)**2)
    
    def mae(self, y, y_pred):
        return np.mean(np.abs(y - y_pred))
    
    def rmse(self, y, y_pred):
        return np.sqrt(self.mse(y, y_pred))
    
    def _set_loss(self):
        name = self.loss_func
        if name == 'mse':
            self.loss = self.mse
            self.grad = self.mse_grad
        elif name == 'mae':
            self.loss = self.mae
            self.grad = self.mae_grad
        else:
            self.loss = self.rmse
            self.grad = self.rmse_grad
    
    def _output_delta(self, output, y):
        n = output.shape[0]
        if self.loss_func == 'mse':
            return 2 * (output - y) / n
        elif self.loss_func == 'mae':
            return np.sign(output - y) / n
        else:
            rmse_val = self.rmse(y, output)
            return (output - y) / (n * rmse_val)

    def forward(self, X):
        self.cache = []
        input = X
        
        for i in range(len(self.weights)):
            z = input @ self.weights[i] + self.bias[i]
            a = self.activation(z)
            self.cache.append((z, a))
            input = a
            
        return a

    def backward(self, X, y, output):
        delta = self._output_delta(output, y)

        for i in reversed(range(len(self.weights))):
            if i == 0:
                a_prev = X
            else:
                a_prev = self.cache[i-1][1]

            w = self.weights[i]
            b = self.bias[i]
            wd = self.weight_decay
            lr = self.learning_rate

            dw = a_prev.T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            
            self.weights[i] = (1 - wd * lr) * w - lr * dw
            self.bias[i] =  (1 - wd * lr) * b - lr * db

            if i > 0:
                z_prev = self.cache[i-1][0]
                delta = delta @ self.weights[i].T * self.activation_grad(z_prev)


    def fit(self, X_train, y_train, X_val=None, y_val=None):
        return 0
    
    def predict(self, X):
        return 0