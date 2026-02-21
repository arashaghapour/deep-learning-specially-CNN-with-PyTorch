import argparse
import torch
import get_data ,data_setup, engine, model_builder, utils, predict, logger
from pathlib import Path
def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Training Script")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_units", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--download", type=bool, default=False)

    return parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
args = get_args()
path = Path("./data")
train_data = "./data/pizza_steak_sushi/train"
test_data = "./data/pizza_steak_sushi/test"
get_data.get_data(path=Path("./data"), folder_name="pizza_steak_sushi", file_name="pizza_steak_sushi.zip", link="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")

big_train_data = "./data/pizza_steak_sushi_20_percent/train"
big_test_data = "./data/pizza_steak_sushi_20_percent/test"
get_data.get_data(path=Path("./data"), folder_name="pizza_steak_sushi_20_percent", file_name="pizza_steak_sushi_20_percent.zip", link="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip")


model_0, b0_transform = model_builder.create_b0_model(device=device)
model_1, b0_transform = model_builder.create_b2_model(device=device)


train_dataloader, test_dataloader, classes = data_setup.data_to_dataloaders(train_data_path=train_data,
                                                                            test_data_path=test_data,
                                                                            transform=b0_transform,
                                                                            batch_size=args.batch_size)

big_train_dataloader, big_test_dataloader, big_classes = data_setup.data_to_dataloaders(train_data_path=big_train_data,
                                                                                 test_data_path=big_test_data,
                                                                                 transform=b0_transform,
                                                                                 batch_size=args.batch_size)

food101_train_dataset, food101test_dataloader, food101_classes = data_setup.fake_data(transform=b0_transform,
                                                                                      batch_size=args.batch_size)

optimizer = torch.optim.Adam(params=model_0.parameters(), lr=args.lr)
loss_fn = torch.nn.CrossEntropyLoss()
models = {"model_0": model_0}
data = {"train_data": train_dataloader}
model_builder.model_summary(model=model_1)

epochs = [5]
experiment = 0
for model_name, model in models.items():
    for data_name, data in data.items():
        for epoch in epochs:
            print(f"experiment: {experiment}")
            print(f"model_name: {model_name}")
            print(f"data_name: {data_name}")
            print(f"epoch: {epoch}")
            writer = logger.create_writer(model_name=model_name, experiment_name=str(experiment))
            if data_name == "train_data":
                model_builder.adjust_num_classes(model=model, num_class=len(classes), device=device)
            else:
                model_builder.adjust_num_classes(model=model, num_class=len(big_classes), device=device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
            engine.train_model(model=model,
                               device=device,
                               train_data=data,
                               test_data=test_dataloader,
                               epochs=epoch,
                               optimizer=optimizer,
                               loss_fn=loss_fn,
                               writer=writer)
            experiment += 1
            utils.save_model(model=model, path="./models")
            


food101_train_dataloader, food101test_dataloader, food101_classes = get_data.get_foof101dataset(root=path/"food101",
                                                                                             transform=b0_transform,
                                                                                             download=args.download,
                                                                                             batch_size=args.batch_size)

import os
os.mkdir()
models = {"model_0": model_0}
lrs = [0.1]
hidden_units = [10]
model_builder.adjust_b0_num_classes(model=model_0, num_class=len(food101_classes), device=device)
model_builder.adjust_num_classes(model=model_1, num_class=len(food101_classes), device=device)
for model_name, model in models.items():
    for lr in lrs:
        for hidden_unit in hidden_units:
            print(f"experiment: {experiment}")
            print(f"learning_rate: {lr}")
            print(f"hidden_unit: {hidden_unit}")
            print(f"model: {model_name}")
            writer = logger.create_writer(model_name=model_name, experiment_name=str(experiment))
            engine.train_model(model=model,
                               device=device,
                               train_data=food101_train_dataloader,
                               test_data=food101test_dataloader,
                               epochs=epoch,
                               optimizer=optimizer,
                               loss_fn=loss_fn,
                               writer=writer)
            experiment += 1

