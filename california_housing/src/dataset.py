import polars as pl
from sklearn.datasets import fetch_california_housing

class Data:
    def __init__(self):
        data = fetch_california_housing().data
        self.df = pl.DataFrame(data)
        
    def load(self):
        return self.df