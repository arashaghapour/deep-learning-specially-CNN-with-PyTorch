from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


def create_writer(model_name: str,
                  experiment_name: str):
    log_dir = os.path.join("./run", model_name, experiment_name)
    return SummaryWriter(log_dir=log_dir)