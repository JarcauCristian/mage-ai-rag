
import pandas as pd
from mage_ai.data_preparation.decorators import transformer, test

@transformer
def transform_data_filter_rows(*args, **kwargs):
    df = args[0]
    df_filtered = df[df['existing_column'] > 10]
    return df_filtered

@test
def test_transform_data_filter_rows(*args) -> None:
    df = pd.DataFrame({'existing_column': [5, 15, 25]})
    df_transformed = transform_data_filter_rows(df)
    assert len(df_transformed) == 2
    