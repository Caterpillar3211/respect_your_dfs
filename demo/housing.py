import os
from context import rdfs as rd

from sklearn.pipeline import Pipeline

df = rd.load_dataset(file=os.path.join(
    os.path.dirname(__file__), 'housing_dataset.csv'))

num_columns = list(df.drop('ocean_proximity', axis=1))
cat_columns = ['ocean_proximity', ]
cat_values = [df['ocean_proximity'], ]

pipe = Pipeline([
                ('imputer', rd.Imputer(strategy='median')),
                ('merger', rd.Merger(cat_columns, cat_values)),
                ('category_encoder', rd.CategoryEncoder(cat_columns)),
                ])

df_tr = pipe.fit_transform(df[num_columns])
