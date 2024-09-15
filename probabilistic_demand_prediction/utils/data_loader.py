import json
import zipfile
import pandas as pd
import requests
import numpy as np


# Utility functions for specific data
def get_dummies(series: pd.Series, column_names: list):
    dummy_df = pd.get_dummies(series).astype(np.int32)
    if len(column_names) == dummy_df.shape[-1]:
        dummy_df.columns = column_names
    else:
        raise ValueError("Length of column names list does not match the number of dummy variables")
        return pd.DataFrame(data=None)
    return dummy_df


def is_weekend(day: int):
    if day in [6, 7]:
        return 1
    return 0


def is_work_hour(day: int, hour: int):
    if not is_weekend(day):
        if 8 <= hour <= 17:  # LBNL runs between 8 AM to 5 PM on weekdays
            return 1
    return 0


def standardize(y: pd.Series):
    y_mean = y.mean()
    y_std = y.std()
    y = (y - y_mean) / y_std
    return y, y_mean, y_std


class Dataset:
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        # Read the json and store
        try:
            f = open(self.json_file_path, "r")
            self.datasets = json.load(f)
        except Exception as ex:
            print(ex)

    def _download(self, url: str, path_type: str):
        if path_type == "local":
            # filename = url.split("/")[-1]
            ref = zipfile.ZipFile(url, "r")
            return ref
        elif path_type == "internet":
            response = requests.get(url)
            if response.status_code == 200:
                filename = url.split("/")[-1]
                file = open(filename, "wb")
                file.write(response.content)
                file.close()
                ref = zipfile.ZipFile(filename, "r")
                return ref
            else:
                raise FileNotFoundError("URL does not contain a ZIP file or the specified path cannot be found")

    def get_data(self, dataset_name: str):
        dataset = {}
        path_type = self.datasets[dataset_name]["pathtype"]
        print(f"Loading dataset from source: {path_type}")
        ref = self._download(self.datasets[dataset_name]["url"], path_type)
        target_names = self.datasets[dataset_name]["target_names"]
        fullpath = {}
        for f in ref.namelist():
            for t in target_names:
                if t in f:
                    fullpath[t] = f

        for t, path in fullpath.items():
            f = ref.open(path, "r")
            df = pd.read_csv(f)
            dataset[t] = df
        return dataset

    def load_time_series(self, df: pd.DataFrame, x_columns: list, y_columns: list,
                         start_train: str, end_train: str, start_test: str, end_test: str,
                         forecast_horizon: np.int32, look_back_size: np.int32):

        start_train = pd.to_datetime(start_train)
        end_train = pd.to_datetime(end_train)

        x = df[x_columns].copy()
        y = df[y_columns].copy()

        start_train = pd.to_datetime(start_train)
        end_train = pd.to_datetime(end_train)
        x_train = x[start_train:end_train]
        y_train = y[start_train:end_train]

        train_index = y_train.index

        timediff = forecast_horizon * np.ones(shape=(y_train.shape[0],))
        for i, diff in enumerate(timediff[:forecast_horizon + 1]):
            timediff[i] = min(diff, i)

        latest_idx = train_index - pd.to_timedelta(timediff, unit='H')
        latest_measurement_train = y_train.loc[latest_idx].copy()
        latest_measurement_train.index = train_index
        latest_measurement_train = latest_measurement_train.add_prefix('latest_')
        x_train = pd.concat([latest_measurement_train, x_train], axis=1)

        start_test = pd.to_datetime(start_test) - pd.to_timedelta(look_back_size, unit='H')
        end_test = pd.to_datetime(end_test)
        y_test = y[start_test:end_test].copy()
        test_index = y_test.index
        x_test = x[start_test:end_test].copy()

        timediff = forecast_horizon * np.ones(shape=(y_test.shape[0],))
        for i, diff in enumerate(timediff[:forecast_horizon + 1]):
            timediff[i] = min(diff, i)

        latest_idx = test_index - pd.to_timedelta(timediff, unit='H')
        latest_measurement_test = y_test.loc[latest_idx].copy()
        latest_measurement_test.index = test_index
        latest_measurement_test = latest_measurement_test.add_prefix('latest_')
        x_test = pd.concat([latest_measurement_test, x_test], axis=1)

        x_train = x_train.values
        y_train = y_train.values
        x_test = x_test.values
        y_test = y_test.values

        return x_train, y_train, x_test, y_test
