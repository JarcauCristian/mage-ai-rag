
import pandas as pd
from mage_ai.data_preparation.decorators import transformer, test

@transformer
def transform_data_convert_dtypes(*args, **kwargs):
    df = args[0]
    df['existing_column'] = df['existing_column'].astype(float)
    return df

@test
def test_transform_data_convert_dtypes(*args) -> None:
    df = pd.DataFrame({'existing_column': ['1', '2', '3']})
    df_transformed = transform_data_convert_dtypes(df)
    assert df_transformed['existing_column'].dtype == float
    