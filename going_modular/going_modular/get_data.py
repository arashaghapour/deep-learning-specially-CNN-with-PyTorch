import zipfile, requests, os
from pathlib import Path
import pathlib
def get_data(path: pathlib.WindowsPath):
    """
    get_data function give path and if data data don`t exit download it
    Args:
        path: path of data or where you want download data

    Returns:
        just print what happened
        Example:
            get_data(./data)
    """
    path1 = path / "pizza_sushi_steak"
    if path1.is_dir():
        print("data already exits")
    else:
        zip_path = path/"pizza_steak_sushi.zip"
        if not zip_path.is_file():
            with open(zip_path, mode="wb") as f:
                print("downoaling...")
                req = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
                f.write(req.content)
        with zipfile.ZipFile(zip_path, mode="r") as f:
            print("extracking...")
            f.extractall(path1)
        zip_path.unlink()

