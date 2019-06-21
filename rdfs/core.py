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



class AttributeAdder(TransformerMixin):

    """
    AttributeAdder object 

    It is used to add new columns (features) to the dataframe.

    Methods:
        __init__(name, function, parameters) <- 'name' is the label of new column, 
                                                'function' is a function upon which the values will be created,
                                                'parameters' is a list of column names (str) and/or constant parameters.

        new_attribute(name, function, parameters) <- ...

        fit(X, y) <- returns itself.

        transform(X, y) <- performs the transformation (adds new attribute) on the dataframe and returns it.

        fit_transform(X, y) <- combined  fit(X, y)  and  transform(X, y). 
                               This is advised in most cases just to stay friendly with sklearn module.

    """

    def __init__(self, name, function, parameters):
        self.name = name
        self.function = function
        self.parameters = parameters

    def new_attribute(self, name, function, parameters):
        self.name = name
        self.function = function
        self.parameters = parameters

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        parameters = []
        for parameter in self.parameters:
            if isinstance(parameter, str):
                parameter = X[parameter]
            parameters.append(parameter)
        X[self.name] = self.function(*parameters)
        return X


class Pipesystem(TransformerMixin):

    """
    Pipesystem object 

    It works (it has less features though) as  sklearn.Pipeline  .

    Methods:
    
        __init__(verbose[, False]) <- if 'verbose' is True, then everytime a transformation is made,
                                      it will print out the information about it.

        new_pipe(pipe_set, always_active[, True]) <- creates a new pipe (ordered),
                                                     pipe_set is expected to be a tuple of name and object
                                                     ( in that order ). always_active  does not have any functionality
                                                     at this moment. It is expected for it to be a indicator for automatic
                                                     dataframe modeling for best predictions later on.
    
        show_pipeline() <- returns an ordered list with all current pipes.

        fit(X, y) <- returns itself.

        transform(X, y) <- performs all transformations (from all pipes) on the dataframe and returns it.

        fit_transform(X, y) <- combined  fit(X, y)  and  transform(X, y). 
                               This is advised in most cases just to stay friendly with sklearn module.

    """

    def __init__(self, verbose=False):
        self._pipes = []
        self._activated = {}
        self._verbose = verbose

    def new_pipe(self, pipe_set, always_active=True):
        name, pipe = pipe_set
        self._pipes.append((name, pipe))
        self._activated[name] = True

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for name, pipe in self._pipes:
            if self._activated[name] == False:
                continue
            if self._verbose:
                print(f'> pushing through \'{name}\' with {pipe}')
            X = pipe.fit_transform(X)
        return X

    def show_pipeline(self):
        out = []
        for name, _ in self._pipes:
            if self._activated[name]:
                out.append(name)
        return out

    def _activate_array(self, array):
        for value, name, _ in zip(array, self._pipes):
            if not value:
                self._disable_pipe(name)

    def _disable_pipe(self, name):
        self._activated[name] = False


class OptimizedPipesystem(Pipesystem):

    """
    OptimizedPipesystem object 

    Enhanced  rdfs.Pipesystem  , it uses one of the optimiztion methods to determine
    the most promising features without actually training a model.

    One way of optimization (and currently, the only one implemented) is correlation.
    Upon object creation, specify  optimization  parameter to  'corr_<int>',
    the integer will be the percent rate from 0 to 100 and will act like a filter,
    every feature that is less significant than that, will not be a part of returned dataframe.

    Methods:

        __init__(optimize_for, optimization[, 'corr_20'], verbose[, False]) <- optimize_for  (str) are the target columns (labels).
                                                                               optimization  (str) is the method used to optimize.
                                                                               If 'verbose' is True, then everytime a transformation is made,
                                                                               it will print out the information about it.

        new_pipe(pipe_set, always_active[, True]) <- creates a new pipe (ordered),
                                                     pipe_set is expected to be a tuple of name and object
                                                     ( in that order ). always_active  does not have any functionality
                                                     at this moment. It is expected for it to be a indicator for automatic
                                                     dataframe modeling for best predictions later on.
    
        show_pipeline() <- returns an ordered list with all current pipes.

        fit(X, y) <- returns itself.

        transform(X, y) <- performs all transformations (from all pipes) on the dataframe, chooses the 
                           most meaningful features and returns the dataframe.

        fit_transform(X, y) <- combined  fit(X, y)  and  transform(X, y). 
                               This is advised in most cases just to stay friendly with sklearn module.

    """

    def __init__(self, optimize_for, optimization='corr_20', verbose=False):
        Pipesystem.__init__(self, verbose)
        self._target = optimize_for
        self._optimization = optimization
        self._best_parameters = []

    def transform(self, X, y=None):
        X = Pipesystem.transform(self, X, y)
        opt = getattr(self, '_optimization', 'corr_20')

        if opt[:5] == 'corr_':
            threshold = int(opt[5:]) / 100
            corr_table = X.corr()[getattr(self, '_target')].sort_values(ascending=False).to_dict()
            
            self._best_parameters = [name for name in corr_table if abs(corr_table[name]) >= threshold]

        return X[self._best_parameters]