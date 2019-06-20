import os
import pandas as pd
import numpy as np

from context import rdfs

from rdfs import DFTransformer
from rdfs import Merger
from rdfs import CategoryEncoder
from rdfs import load_dataset

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline



df = load_dataset(file=os.path.join(os.path.dirname(__file__), 'housing_dataset.csv'))

num_columns = list(df.drop('ocean_proximity', axis=1))
cat_columns = ['ocean_proximity']

transformer = DFTransformer(num_columns)
merger = Merger(cols_names=['ocean_proximity'], cols_values=[df['ocean_proximity'].values])
imputer = SimpleImputer(strategy='median')
category_encoder = CategoryEncoder(merger, cat_columns)

full_pipeline = Pipeline([
        ('df_to_array', transformer),
        ('imputer', imputer),
        ('array_to_df', transformer),
        ('merge_with_categories', merger),
        ('categories_encoder', category_encoder),
    ])

df_tr = full_pipeline.fit_transform(df)
print(df_tr)