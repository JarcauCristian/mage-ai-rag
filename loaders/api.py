import io
import pandas as pd
import requests

@data_loader
def load_data_from_api(*args, **kwargs):
    """
    Template for loading data from API
    """
    url = ''
    response = requests.get(url)

    return pd.read_csv(io.StringIO(response.text), sep=',')
