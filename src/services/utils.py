import kagglehub
from os import path, mkdir, rename, listdir


def get_data(data_handle: str, data_folder: str) -> str | None:
    if not path.exists("data"):
        mkdir("data")

    data_output_folder = path.join("data", data_folder)
    if path.exists(data_output_folder) and listdir(data_output_folder):
        print(f"Files in output path '{data_output_folder}':\n{listdir(data_output_folder)}\n"
              f"Please clean if unintended.")
        return None
    else:
        # TODO: for some reason kaggle download doesn't save to path so have to manually move
        data_path = kagglehub.dataset_download(handle=data_handle)
        rename(data_path, data_output_folder)
        print(f"Dataset {data_handle} downloaded to {data_output_folder}")
        return data_output_folder
