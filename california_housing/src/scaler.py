import numpy as np

class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None 

    def fit(self, x):
        x = x.to_numpy()
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.std[self.std == 0] = 1e-8
        return self 
    
    def transform(self, x):
        x = x.to_numpy()
        return (x - self.mean) / self.std
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)