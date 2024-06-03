import os
import pickle
# import click
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


# @click.command()
# @click.option(
#     "--data_path",
#     default="./output",
#     help="Location where the processed NYC taxi trip data was saved"
# )
def run_train(data_path: str):
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        mlflow.log_param("train-data-path", f"{os.path.join(data_path, 'train.pkl')}")
        mlflow.log_param("test-data-path", f"{os.path.join(data_path, 'test.pkl')}")

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mlflow.log_metric("rmse", rmse)


# if __name__ == '__main__':
    
#     run_train()
