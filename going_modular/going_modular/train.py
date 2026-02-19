import argparse
import torch
import get_data, data_to_tensor, engine, model_builder, utils, predict
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
train_data = "./data/pizza_sushi_steak/train"
test_data = "./data/pizza_sushi_steak/test"
get_data.get_data(Path("./data"))
train_dataloader, test_dataloader, class_names = data_to_tensor.data_to_dataloaders(train_data_path=train_data,
                                       test_data_path=test_data,
                                       batch_size=args.batch_size)
model_0 = model_builder.TinyVGG(input_shape=3,
                      output_shape=10,
                      hidden_units=args.hidden_units).to(args.device)
# optimizer = torch.optim.Adam(params=model_0.parameters(), lr=args.lr)
# loss_fn = torch.nn.CrossEntropyLoss()
# engine.train_model(model=model_0,
#                    device=args.device,
#                    train_data=train_dataloader,
#                    test_data=test_dataloader,
#                    epochs=args.epochs,
#                    optimizer=optimizer,
#                    loss_fn=loss_fn)
# utils.save_model(model=model_0, path="./models/model.pth")
utils.load_model(model=model_0, model_path="./models/model.pth")
train_dataset, test_dataset = data_to_tensor.data_to_dataset(train_data_path=train_data, test_data_path=test_data)
predict.predict(model=model_0, test_data=test_dataset, num_pic=5)

