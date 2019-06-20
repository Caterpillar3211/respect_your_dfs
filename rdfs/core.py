import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from .helpers import encoded_array_to_df_compatible_array
from .base import ShapeException, NotADataFrameException, Transformer


class Merger(TransformerMixin):

    """ 
    Merger Object

    It is used to merge dataframes with given columns.
    It is probably useful only for pipelines, as you
    can easily achieve the same result with basic pandas operations.

    Unlike other objects, it does not inherit Transformer class,
    as it doesn't need to transform dataframe to array or vice-versa.

    You can specify column_names and column_values upon creating the object,
    or call  'new_merge'  method with those parameters.

    If X parameter of transformation is not a dataframe, raises an exception.
    """

    def __init__(self, cols_names=None, cols_values=None):
        self._cols_names = cols_names
        self._cols_values = cols_values

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise NotADataFrameException
        for name, values in zip(self._cols_names, self._cols_values):
            X[name] = values
        return X

    def new_merge(self, cols_names, cols_values):
        self._cols_names = cols_names
        self._cols_values = cols_values
        return self


class CategoryEncoder(Transformer, TransformerMixin):

    """ 
    CategoryEncoder object

    Upon creation, you should specify column names that will be encoded.
    Alternatively you can set them with set_columns method, or display them
    with get_columns method.

    It is used to encode categorical attributes of the dataframe.
    It contains it's merger  _merger  , as well as specified encoder  _encoder  .

    Possible encodings:
        - 'onehot'
    """

    def __init__(self, columns, encoder='onehot', encoder_params=[]):
        self._columns = columns
        self._merger = Merger()
        self._encoder_type = encoder
        if self._encoder_type == 'onehot':
            self._encoder = OneHotEncoder(*encoder_params)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise NotADataFrameException
        if self._encoder_type == 'onehot':
            values = self._encoder.fit_transform(X[self._columns]).toarray()
            values = encoded_array_to_df_compatible_array(values)
            features = self._encoder.get_feature_names()
            features = [feature[3:] for feature in features]
            return self._merger.new_merge(features, values).fit_transform(X.drop(self._columns, axis=1), y)


class Imputer(Transformer, TransformerMixin):

    """ 
    Imputer object

    It is a wrapper around sklearn.impute.SimpleImputer,
    all it does, is that it takes dataframe as an input, which is transformed
    into np.ndarray, fed into actual SimpleImputer object, and the result is returned
    as a dataframe, with the same exact columns.
    """

    def __init__(self, missing_values=np.nan, strategy='mean', fill_value=None, verbose=0, copy=True, add_indicator=False):
        self._imputer = SimpleImputer(
            missing_values, strategy, fill_value, verbose, copy, add_indicator)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        if isinstance(X, pd.DataFrame):
            self._columns = X.columns
            X_tr = self._imputer.fit_transform(self._to_array(X))
            X_tr = self._to_df(X_tr)
            return X_tr
        else:
            return self._imputer.fit_transform(X, y)
