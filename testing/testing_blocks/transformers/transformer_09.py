from mage_ai.data_preparation.decorators import transformer, test


@transformer
def transform_data_remove_duplicates(data, *args, **kwargs):
    data = data.drop_duplicates()
    return data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
