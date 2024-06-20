
import pandas as pd
from mage_ai.data_preparation.decorators import transformer, test

@transformer
def transform_data_drop_columns(*args, **kwargs):
    df = args[0]
    df = df.drop(columns=['unwanted_column'])
    return df

@test
def test_transform_data_drop_columns(*args) -> None:
    df = pd.DataFrame({'unwanted_column': [1, 2, 3], 'wanted_column': [4, 5, 6]})
    df_transformed = transform_data_drop_columns(df)
    assert 'unwanted_column' not in df_transformed.columns
    