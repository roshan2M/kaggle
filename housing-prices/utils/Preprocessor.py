from DataFrameImputer import DataFrameImputer
import pandas as pd
import numpy as np

class Preprocessor(object):
    def __init__(self):
        """Class to preprocess input dataframe.

        Loading data, inputting missing values, transforming columns.
        """
        pass

    def load_data(self, file):
        return pd.read_csv(file, sep=',')

    def fill_missing_values(self, df):
        return DataFrameImputer().fit_transform(df)
