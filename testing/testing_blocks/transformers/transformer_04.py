
import pandas as pd
from mage_ai.data_preparation.decorators import transformer, test

@transformer
def transform_data_rename_column(*args, **kwargs):
    df = args[0]
    df = df.rename(columns={'old_column': 'new_column'})
    return df

@test
def test_transform_data_rename_column(*args) -> None:
    df = pd.DataFrame({'old_column': [1, 2, 3]})
    df_transformed = transform_data_rename_column(df)
    assert 'new_column' in df_transformed.columns
    assert 'old_column' not in df_transformed.columns
    