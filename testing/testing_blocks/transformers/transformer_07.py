
import pandas as pd
from mage_ai.data_preparation.decorators import transformer, test

@transformer
def transform_data_groupby(*args, **kwargs):
    df = args[0]
    df_grouped = df.groupby('group_column').sum().reset_index()
    return df_grouped

@test
def test_transform_data_groupby(*args) -> None:
    df = pd.DataFrame({'group_column': ['A', 'B', 'A'], 'value': [1, 2, 3]})
    df_transformed = transform_data_groupby(df)
    assert len(df_transformed) == 2
    