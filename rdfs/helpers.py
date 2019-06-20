import pandas as pd
import numpy as np

def load_dataset(file):
    return pd.read_csv(file)


def encoded_array_to_df_compatible_array(array):
    new = []
    for index in range(array.shape[1]):
        new.append([])
        for row in array:
            new[index].append(row[index])
    return np.array(new)