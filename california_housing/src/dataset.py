import polars as pl
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, target="target"):
        data = fetch_california_housing()
        df = pl.DataFrame(data.data, schema=data.feature_names)
        self.df = df.with_columns(pl.Series(target, data.target))
        self.target = target
        
    def get_features(self):
        return self.df.select(pl.all().exclude(self.target))
    
    def get_target(self):
        return self.df.select(self.target)
    
    def split(self, test_size=0.2, random_state=42):
            X = self.get_features()
            y = self.get_target()
            return train_test_split(X, y, test_size=test_size, random_state=random_state)