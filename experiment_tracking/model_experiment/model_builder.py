from torch import nn
import torch
import torchvision
import torchinfo


def create_b0_model(device: str):
    """
    tis function get the device and create create efficientnetb0 on the device

    Args:
    device: The device you run your code

    Returns:
        A tuple of model and its default transform
    """

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device=device)

    for param in model.features.parameters():
        param.requires_grad = False

    return model, weights.transforms()

def model_summary(model: torch.nn.Module):
    """
    Docstring for model_summary
    
    Args:
    model: model biult

    Return:
        model informations
    """

    torchinfo.summary(model=model,
                      input_size=(1, 3, 224, 224),
                      col_names=["input_size", "output_size", "num_params", 'trainable'],
                      col_width=20,
                      row_settings=["var_names"])

def adjust_num_classes(model: torch.nn.Module, num_class: int, device: str):

    """

    Args:
    model: model biult
    num_class: number of label kind that data have

    """
    model.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(in_features=1280, out_features=num_class)
    ).to(device=device)
    model_summary(model=model)


def create_b2_model(device: str):
    """
    tis function get the device and create create efficientnetb2 on the device

    Args:
    device: The device you run your code

    Returns:
        A tuple of model and its default transform
    """

    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights).to(device=device)

    for param in model.features.parameters():
        param.requires_grad = False

    return model, weights.transforms()