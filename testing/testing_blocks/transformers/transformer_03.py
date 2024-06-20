from mage_ai.data_preparation.decorators import transformer, test

@transformer
def transform_data_drop_columns(data, *args, **kwargs):
    unwanted_column = kwargs.get('unwanted_column')
    if unwanted_column is not None:
        data = data.drop(columns=[unwanted_column])
    return data

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
