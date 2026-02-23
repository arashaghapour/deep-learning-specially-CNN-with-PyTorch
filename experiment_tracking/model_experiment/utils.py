import torch
def save_model(model: torch.nn.Module, path: str):
    """
    Docstring for save_model
    This function save model parameters in path

    Args:
        model: learned model
        path: destination that pth file save there
    
    Return:
        nothing
    """
    torch.save(obj=model.state_dict(), f=path)
    print("model saved")

def load_model(model: torch.nn.Module, model_path: str):
    """
    Docstring for save_model
    This function load model parameters in path

    Args:
        model: model
        path: destination that pth file saved there
    
    Return:
        nothing
    """
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print("model loaded")