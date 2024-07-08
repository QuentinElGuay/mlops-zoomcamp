from datetime import datetime
import pandas as pd


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
df_input = pd.DataFrame(data, columns=columns)

year = 2023
month = 1
input_file = f's3://nyc-duration/in/{year:04d}-{month:02d}.parquet'

options = {
    'client_kwargs': {
        'endpoint_url': 'http://127.0.0.1:4566'
    }
}

print('Writing inputfile...')
df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

print('Reading outputfile...')
output_file = f's3://nyc-duration/out/{year:04d}-{month:02d}.parquet'
df_output = pd.read_parquet(output_file, storage_options=options)

print(df_output.predicted_duration.sum())