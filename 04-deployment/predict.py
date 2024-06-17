import logging
import pickle
import sys

import numpy as np
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('logger')

categorical = ['PULocationID', 'DOLocationID']

def read_data(year, month):
    
    filename= f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    logger.info('Reading file %s', filename)

    df = pd.read_parquet(filename)

    # create an artificial ride_id column:
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def predict(df):
    logger.info('Batch predictions')

    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    return model.predict(X_val)


def main(year, month):
    
    df = read_data(year, month)
    y_pred = predict(df)

    logger.info('Mean duration prediction: %s', np.mean(y_pred, axis=0))

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['y_pred'] = y_pred

    output_file = f'predictions_{year:04d}-{month:02d}.parquet'

    logger.info('Exporting predictions as parquet file: %s', output_file)
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


if __name__ == '__main__':

    year = int(sys.argv[1])
    month = int(sys.argv[2])

    main(year, month)
