from torch import nn
import torch
class TinyVGG(nn.Module):
    """
    This class is architector of TinyVGG model that has two conv relu conv relu maxpool and classify layer

    Args:
        input_shape: An integer that refer to color channels you have
        ouputshape: An integer that refer to number classe your data have
        hidden_unnits:  An integer that refer to number of norons that every layer have

    Returns:
        A tensor of model predicts and it`s shape = batches, number_of_classes
    """
    def __init__(self, input_shape: int, output_shape: int, hidden_units: int=10):
        super().__init__()
        self.concblock_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.concblock_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classify_layrer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13, out_features=output_shape)
        )
    def forward(self, x: torch.Tensor):
        return self.classify_layrer(self.concblock_2(self.concblock_1(x)))