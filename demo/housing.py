import os
from context import rdfs as rd

from sklearn.pipeline import Pipeline

df = rd.load_dataset(file=os.path.join(
    os.path.dirname(__file__), 'housing_dataset.csv'))

num_columns = list(df.drop('ocean_proximity', axis=1))
cat_columns = ['ocean_proximity', ]
cat_values = [df['ocean_proximity'], ]


pipesystem = rd.OptimizedPipesystem(verbose=False, optimize_for='median_house_value', optimization='corr_10')
pipesystem.new_pipe(('imputer', rd.Imputer(strategy='median')), always_active=True)
pipesystem.new_pipe(('merger', rd.Merger(cat_columns, cat_values)), always_active=True)
pipesystem.new_pipe(('category_encoder_oh', rd.CategoryEncoder(cat_columns)), always_active=True)
pipesystem.new_pipe(('family_member_count_attr_adder', rd.AttributeAdder(
    name='family_member_count', function=lambda a, b: a / b, parameters=('population', 'households'))))
pipesystem.new_pipe(('rooms_over_bedrooms_attr_adder', rd.AttributeAdder(
    name='rooms_over_bedrooms', function=lambda a, b: a / b, parameters=('total_rooms', 'total_bedrooms'))))
pipesystem.new_pipe(('bedrooms_per_house_attr_adder', rd.AttributeAdder(
    name='bedrooms_per_house', function=lambda a, b: a / b, parameters=('population', 'households'))))

df_opt = pipesystem.fit_transform(df[num_columns])
