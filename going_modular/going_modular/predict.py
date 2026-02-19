import torch
import matplotlib.pyplot as plt
def predict(model: torch.nn.modules, test_data: torch.Tensor, num_pic: int):
    """
    This function get data and show its label and model predict on it
    

    Args:
        model: learned model
        test_data: test data
        num_pic: number of pictures want to show it and model predict

    Return:
        show pictures 
        Example: predict(model=model_0, test_data=test_data, num_pic=10)
    """
    plt.figure(figsize=(25, 7))
    r_pic = torch.randint(low=0, high=len(test_data), size=(num_pic, ))
    for i in range(num_pic):
        plt.subplot(1, num_pic, i+1)
        plt.imshow(torch.Tensor.numpy(test_data[r_pic[i]][0].permute(1, 2, 0)))
        plt.title(f"label: {test_data[r_pic[i]][1]}  pred: {torch.argmax(model(test_data[r_pic[i]][0].unsqueeze(dim=0)), dim=1).item()}")
        plt.axis("off")
    plt.show()