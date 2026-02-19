import argparse
import torch
import get_data ,data_setup, engine, model_builder, utils, predict
from pathlib import Path
def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Training Script")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_units", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()

args = get_args()
path = Path("./data")
train_data = "./data/pizza_sushi_steak_20_percent/train"
test_data = "./data/pizza_sushi_steak_20_percent/test"
get_data.get_data(path=Path("./data"), link="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip")

model_0, transform = model_builder.create_model(device=args.device)
train_dataset, test_dataset, classes = data_setup.create_datasets(train_data_path=train_data,
                                                                  test_data_path=test_data,
                                                                  transform=transform)


# train_dataloader, test_dataloader, class_names = data_setup.data_to_dataloaders(train_data_path=train_data,
#                                                                                 test_data_path=test_data,
#                                                                                 batch_size=args.batch_size,
                                                                                # transform=transform)
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=args.lr)
loss_fn = torch.nn.CrossEntropyLoss()
# engine.train_model(model=model_0,
#                    device=args.device,
#                    train_data=train_dataloader,
#                    test_data=test_dataloader,
#                    epochs=args.epochs,
#                    optimizer=optimizer,
#                    loss_fn=loss_fn)
# utils.save_model(model=model_0, path="./models/model.pth")
utils.load_model(model=model_0, model_path="./models/model.pth")
predict.predict(model=model_0,
                image_pathes=test_data,
                num_pic=5,
                transform=transform,
                classes=classes)


