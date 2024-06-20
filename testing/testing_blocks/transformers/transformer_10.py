from mage_ai.data_preparation.decorators import transformer, test

@transformer
def transform_data_replace_values(data, *args, **kwargs):
    old_value = kwargs.get('old_value')
    new_value = kwargs.get('new_value')

    if None not in [old_value, new_value]:
        data = data.replace({old_value: new_value})
    return df

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
