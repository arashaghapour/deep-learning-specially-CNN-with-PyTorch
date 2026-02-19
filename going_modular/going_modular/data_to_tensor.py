import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
def data_to_dataset(train_data_path: str, test_data_path: str):
    data_tranform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    train_dataset = ImageFolder(root=train_data_path,
                                transform=data_tranform)
    test_dataset = ImageFolder(root=test_data_path,
                               transform=data_tranform)
    return train_dataset, test_dataset
def data_to_dataloaders(train_data_path: str, test_data_path: str, batch_size: int):
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
    train_dataset, test_dataset = data_to_dataset(train_data_path=train_data_path, test_data_path=test_data_path)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  pin_memory=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  pin_memory=True)
    return train_dataloader, test_dataloader, train_dataset.classes