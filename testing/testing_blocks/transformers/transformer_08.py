
import pandas as pd
from mage_ai.data_preparation.decorators import transformer, test

@transformer
def transform_data_sort_values(*args, **kwargs):
    df = args[0]
    df = df.sort_values(by='existing_column')
    return df

@test
def test_transform_data_sort_values(*args) -> None:
    df = pd.DataFrame({'existing_column': [3, 1, 2]})
    df_transformed = transform_data_sort_values(df)
    assert (df_transformed['existing_column'].values == [1, 2, 3]).all()
    