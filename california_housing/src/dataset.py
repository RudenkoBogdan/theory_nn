import polars as pl
from sklearn.datasets import fetch_california_housing

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