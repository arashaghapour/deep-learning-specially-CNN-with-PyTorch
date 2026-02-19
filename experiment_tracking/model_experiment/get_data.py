import zipfile, requests, os
from pathlib import Path
import pathlib
import torchvision
from torch.utils.data import DataLoader


def get_data(path: pathlib.WindowsPath, link: str, folder_name: str, file_name: str):
    """
    get_data function give path and if data data don`t exit download it
    Args:
        path: path of data or where you want download data
        link: the link in the internet that data exist there
        
    Returns:
        just print what happened
        Example:
            get_data("./data", "data.com")
    """
    path1 = path/folder_name
    if path1.is_dir():
        print("data already exits")
    else:
        zip_path = path/file_name
        if not zip_path.is_file():
            with open(zip_path, mode="wb") as f:
                print("downoaling...")
                req = requests.get(link)
                f.write(req.content)
                print("file downloaded")
        with zipfile.ZipFile(zip_path, mode="r") as f:
            print("extracking...")
            f.extractall(path1)
            print("file extracked")
        zip_path.unlink()

def get_foof101dataset(root: str, transform, download: bool, batch_size: int):

    """
    This function download the torchvision 101 food dataset and return it.
    
    """
    
    train_dataset = torchvision.datasets.Food101(root=root,
                                                    split="train",
                                                    transform=transform,
                                                    download=download)
    test_dataset = torchvision.datasets.Food101(root=root,
                                                   split="test",
                                                   transform=transform,
                                                   download=download)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    
    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=batch_size)
    return train_dataloader, test_dataloader, train_dataloader.classes