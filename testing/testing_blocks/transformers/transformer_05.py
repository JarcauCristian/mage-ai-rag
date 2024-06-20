from mage_ai.data_preparation.decorators import transformer, test

@transformer
def transform_data_convert_dtypes(data, *args, **kwargs):
    existing_column = kwargs.get('existing_column')
    if existing_column is not None:
        data[existing_column] = data[existing_column].astype(float)
    return data

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
