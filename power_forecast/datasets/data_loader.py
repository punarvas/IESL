import json
import zipfile
import pandas as pd
import requests


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
