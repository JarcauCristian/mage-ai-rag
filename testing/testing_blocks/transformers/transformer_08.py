from mage_ai.data_preparation.decorators import transformer, test

@transformer
def transform_data_sort_values(data, *args, **kwargs):
    existing_column = kwargs.pop('existing_column')
    data = data.sort_values(by=existing_column)
    return data

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
