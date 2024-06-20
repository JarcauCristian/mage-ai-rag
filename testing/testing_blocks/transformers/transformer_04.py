from mage_ai.data_preparation.decorators import transformer, test

@transformer
def transform_data_rename_column(data, *args, **kwargs):
    old_column = kwargs.get('old_column')
    new_column = kwargs.get('new_column')

    if None not in [old_column, new_column]:
        data = data.rename(columns={old_column: new_column})
    return data

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
