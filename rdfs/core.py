import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer as Imputer
from sklearn.pipeline import Pipeline




class ShapeException(Exception):
    """ An exception caused by transformation when 
    the array is out of shape with columns. """
    


class NotADataFrameException(Exception):
    """ An exception occuring when trying to 
    transform something that is not a dataframe. """



class DFTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self._columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            if self._columns:
                # convert to array ( with given columns )
                return X[self._columns].values
            else:
                # convert to array ( with all columns )
                self._columns = X.columns
                return X.values
        elif isinstance(X, np.ndarray):
            if X.shape[1] != len(self._columns):
                print(f'Array out of shape, mismatched columns.\
                        Got {X.shape[1]}, Expected {len(self._columns)}')
                raise ShapeException
            return pd.DataFrame(X, columns=self._columns)

    def fit_transform(self, X, y=None):
        return self.fit(X, y=None).transform(X, y=None)

    @property    
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, columns):
        self._columns = columns



class Merger(BaseEstimator, TransformerMixin):

    def __init__(self, cols_names, cols_values):
        self._columns_names = cols_names
        self._columns_values = cols_values

    def new_merge(self, cols_names, cols_values):
        self._columns_names = cols_names
        self._columns_values = cols_values
        return self

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for values, name in zip(self._columns_values, self._columns_names):
            X[name] = values
        return X



class CategoryEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, merger, categories):
        self._categories = categories
        self.merger = merger
        self.encoder = OneHotEncoder()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise NotADataFrameException
        values = self.encoder.fit_transform(X[self._categories]).toarray()
        values = self.encoded_array_to_df_compatible_array(values)
        features = self.encoder.get_feature_names()
        return self.merger.new_merge(features, values).transform(X.drop(self._categories, axis=1))

    @staticmethod
    def encoded_array_to_df_compatible_array(array):
        new = []
        for index in range(array.shape[1]):
            new.append([])
            for row in array:
                new[index].append(row[index])
        return np.array(new)




