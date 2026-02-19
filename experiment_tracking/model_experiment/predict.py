import torch
import matplotlib.pyplot as plt
import random
import PIL
from pathlib import Path
import data_setup


def predict(model: torch.nn.Module, image_pathes: str, num_pic: int, transform, classes):
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