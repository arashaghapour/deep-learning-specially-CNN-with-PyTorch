import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision

def imagefolder_without_transform(data_path: str):
    dataset = ImageFolder(root=data_path)

    return dataset



def create_datasets(train_data_path: str, test_data_path: str, transform):
    """
    This function create a dataset of data with train and test path given and transform

    Args:
        train_data_path: string of the train data path
        test_data_path: string of the train data path
        transform: should pretrained model main transform

    Returns:
        tuple of train_dataset and test_dataset and data classes
    """
    train_dataset = ImageFolder(root=train_data_path,
                                transform=transform)

    test_dataset = ImageFolder(root=test_data_path,
                               transform=transform)
    
    return train_dataset, test_dataset, test_dataset.classes


def data_to_dataloaders(train_data_path: str, test_data_path: str, batch_size: int, transform):
    """
    give this function train and test paths and gives you dataloader format of the data
    Args:
        train_data_path: train path and data folders must be in standard format
        test_data_path: tEST path and data folders must be in standard format
        batch_size: must be integer and its adjust the number picture per batch

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names)
        where class_names is a list of labels
        Example usage:
            train_dataloader, test_dataloader, claas_names = data_to_data_loader(train_path="./data/train",
            test_path="./data/test",
            batch_size=10)
    """
    train_dataset, test_dataset, classes = create_datasets(train_data_path=train_data_path,
                                                           test_data_path=test_data_path,
                                                           transform=transform)
    
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size)
    
    return train_dataloader, test_dataloader, train_dataset.classes