import torch
import matplotlib.pyplot as plt
import random
import PIL
from pathlib import Path
import data_setup


def predict(model: torch.nn.Module, image_pathes: str, num_pic: int, transform, classes):
    """
    This function make predictions with given model and given image pathes in standard form of the data imagefolder

    Args:
        mdoel: the model trained
        image_pathes: the standard form of the data path
        num_pic: number of the pictures given to predim with model
        transform: the standard model tranform to fit the data with model input
        classes: the number of the classes data have

    Return:
        show pictures and their labels and model predict on them
    """
    images = data_setup.imagefolder_without_transform(image_pathes)
    r_images = torch.randint(low=0, high=len(images), size=(num_pic, ))
    plt.figure(figsize=(15, 7))
    model.eval()
    with torch.inference_mode():
        for i in range(num_pic):
            img, label = images[r_images[i]]
            tranform_img = transform(img)
            plt.subplot(1, num_pic, i+1)
            plt.title(f"lable: {label}, model pred: {classes[model(tranform_img.unsqueeze(dim=0)).argmax(dim=1).item()]}")
            plt.imshow(img)
            plt.axis(False)
        plt.show()