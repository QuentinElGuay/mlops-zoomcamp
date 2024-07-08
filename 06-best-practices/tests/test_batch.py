from datetime import datetime

import pandas as pd
import pytest

from batch import prepare_data


def test_prepare_data():

    def dt(hour, minute, second=0):
        return datetime(2023, 1, 1, hour, minute, second)
    
    categorical = ['PULocationID', 'DOLocationID']

    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]
    columns = [
        'PULocationID',
        'DOLocationID',
        'tpep_pickup_datetime',
        'tpep_dropoff_datetime'
    ]
    df = pd.DataFrame(data, columns=columns)

    expected_data = [
        ('-1', '-1', dt(1, 1), dt(1, 10), 9.0),
        ('1', '1', dt(1, 2), dt(1, 10), 8.0)
    ]
    columns.append('duration')
    expected_df = pd.DataFrame(expected_data, columns=columns)
    
    result_df = prepare_data(df, categorical)

    pd.testing.assert_frame_equal(expected_df, result_df)
