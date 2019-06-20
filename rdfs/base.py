import pandas as pd
import numpy as np


class ShapeException(Exception):
    """ An exception caused by transformation when 
    the array is out of shape with columns. """


class NotADataFrameException(Exception):
    """ An exception occuring when a dataframe is needed, 
    but not received. """


class Transformer:

    """ Base class for any object that has to perform transformations on a dataframe. """

    _columns = None

    def _to_array(self, df):
        """ 
        Transforms given pd.DataFrame to np.ndarray, 
        containing columns that were specified in the object.

        If columns weren't specified, then returns the whole dataframe as an array,
        and saves it's columns for later.
        """

        if self._columns is not None:
            # convert to array ( with given columns )
            return df[self._columns].values
        else:
            self._columns = df.columns
            # convert to array ( with all columns )
            return df.values

    def _to_df(self, array):
        """ 
        Transforms given np.ndarray to pd.DataFrame,
        assigning it columns that were specified in the object.

        If columns weren't specified, then returns unconverted array (as passed).
        If columns are out of shape with array columns, raises a ShapeException.
        """
        if self._columns is None:
            print(
                self, '-> no columns specified when trying to convert an array to dataframe.')
            return array
        if array.shape[1] != len(self._columns):
            print(
                self, f'-> array out of shape with columns, expected ({len(self._columns)}), got ({X.shape[1]})')
            raise ShapeException

        return pd.DataFrame(array, columns=self._columns)

    def set_columns(self, columns):
        self._columns = columns

    def get_columns(self):
        return self._columns
