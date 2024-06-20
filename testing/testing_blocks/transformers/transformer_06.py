from mage_ai.data_preparation.decorators import transformer, test

@transformer
def transform_data_fill_na(data, *args, **kwargs):
    data = data.fillna(0)
    return data

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
    