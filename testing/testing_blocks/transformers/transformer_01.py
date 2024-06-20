
import pandas as pd
from mage_ai.data_preparation.decorators import transformer, test

@transformer
def transform_data_add_column(*args, **kwargs):
    df = args[0]
    df['new_column'] = df['existing_column'] * 2
    return df

@test
def test_transform_data_add_column(*args) -> None:
    df = pd.DataFrame({'existing_column': [1, 2, 3]})
    df_transformed = transform_data_add_column(df)
    assert 'new_column' in df_transformed.columns
    assert (df_transformed['new_column'] == df['existing_column'] * 2).all()
    