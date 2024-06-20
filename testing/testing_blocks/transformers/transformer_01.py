from mage_ai.data_preparation.decorators import transformer, test

@transformer
def transform_data_add_column(data, *args, **kwargs):
    new_column = kwargs.get("new_column")
    existing_column = kwargs.get("existing_column")
    if None not in [new_column, existing_column]:
        data[new_column] = data[existing_column] * 2
    return data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'