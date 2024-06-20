
import pandas as pd
from mage_ai.data_preparation.decorators import transformer, test

@transformer
def transform_data_fill_na(*args, **kwargs):
    df = args[0]
    df = df.fillna(0)
    return df

@test
def test_transform_data_fill_na(*args) -> None:
    df = pd.DataFrame({'existing_column': [1, None, 3]})
    df_transformed = transform_data_fill_na(df)
    assert df_transformed['existing_column'].isna().sum() == 0
    