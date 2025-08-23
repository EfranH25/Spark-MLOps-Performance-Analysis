import kagglehub
from os import path, mkdir, rename, listdir
from shutil import rmtree


DATA_OPTIONS = {"creditcardfraud": "mlg-ulb/creditcardfraud"}


def get_data(
    data_name: str,
):
    if not path.exists("data"):
        mkdir("data")

    if data_name in DATA_OPTIONS:
        if path.exists(path.join("data", data_name)):
            print(f"{data_name} already exists, skipping.")
            return
        data_path = kagglehub.dataset_download(
            handle=DATA_OPTIONS[data_name]
        )  # TODO: for some reason kaggle download doesn't save to path so have to manually move
        rename(data_path, path.join("data", data_name))
        print(f"Dataset {data_name} downloaded.")
    else:
        print(
            f"Dataset {data_name} not found in data options {list(DATA_OPTIONS.keys())}"
        )


def cleanup_data(data_name: str | None = None, clean_all: bool = False):
    if clean_all:
        print("Remove all datasets in 'data'")
        folder_list = listdir("data")
        for folder in folder_list:
            rmtree(path.join("data", folder))
        print("Complete")
    elif data_name:
        if data_name in DATA_OPTIONS:
            rmtree(path.join("data", data_name))
            print(f"Dataset {data_name} removed.")
        else:
            print(
                f"Dataset {data_name} not found in data options {list(DATA_OPTIONS.keys())}. Nothing to clean."
            )
