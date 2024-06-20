from mage_ai.data_preparation.decorators import transformer, test

@transformer
def transform_data_groupby(data, *args, **kwargs):
    group_column = kwargs.get('group_column')
    if group_column is not None:
        data = data.groupby(group_column).sum().reset_index()
    return data

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
    