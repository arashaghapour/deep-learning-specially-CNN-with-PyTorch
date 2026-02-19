import torch


def train_step(model: torch.nn.Module,
               device: str,
               train_data: torch.Tensor,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module) -> tuple[int, int]:
    """
    Docstring for train_step
    This functrion do train step like (do the forward pass, calculate the loss, optimizer zero grad loss backward and optimzer step)
    to fit the model parameters with the data for every bathes that data we have


    Args:
        model: model architeture you deighned for learning
        device: the device you have cuda or cpu
        train_data: train data you have type must DataLoader
        optimizer: optimizer function you want
        loss_fn: loss unction to measure how wrong model is


    Return:
        A tuple of train_loss and train_accuracy
        Example train_loss, train_acc = train_step(model=TinyVGG,
        train_data=train_data,
        optimizer=optimizer,
        loss_fn=loss_fn)
    """
    train_loss, train_acc = 0, 0
    for (X, y) in train_data:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        train_loss += loss.item()
        train_acc += ((torch.argmax(logits, dim=1) == y).sum().item()) / len(y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss / len(train_data), train_acc / len(train_data)


def test_step(model: torch.nn.Module,
              device: str,
              test_data: torch.Tensor,
              loss_fn: torch.nn.Module) -> tuple[int, int]:
    """
    Docstring for train_step
    This functrion do test step like (do the forward pass, calculate the loss)
    to evaluate have model work on data that never see them


    Args:
        model: model architeture you deighned for learning
        device: the device you have cuda or cpu
        train_data: train data you have type must DataLoader
        loss_fn: loss unction to measure how wrong model is

    Return:
        A tuple of test_loss and test_accuracy
        Example test_loss, test_acc = test_step(model=TinyVGG,
        train_data=train_data,
        loss_fn=loss_fn)
    """
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for (X, y) in test_data:
            X, y = X.to(device), y.to(device)
            test_logits = model(X)
            t_loss = loss_fn(test_logits, y)
            test_loss += t_loss
            test_acc += ((torch.argmax(test_logits, dim=1) == y).sum().item()) / len(y)
    return test_loss / len(test_data), test_acc / len(test_data)


def train_model(model: torch.nn.Module,
                device: str,
                train_data: torch.Tensor,
                test_data: torch.Tensor,
                epochs: int,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module):
    """
    This functio learn the model on train data you gave to so for any epoch it 
    learns the patterns in data with two (train_step, test_step) functio we made before
    and return losses and accuracy on trai and test data for each epoch
    
    Args:
        model: modeel architecture you made
        device: device you have like cuda or cpu
        train_data: train_data must be DataLoader
        test_data: test_data must be DataLoader
        epochs: how many times model learn on data
        optimizer: optimizer function you want
        loss_fn: loss unction to measure how wrong model is
    
    Returns:
        print  train test loss and accuracy 
        
    """
    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = train_step(model=model,
                                           device=device,
                                           train_data=train_data,
                                           optimizer=optimizer,
                                           loss_fn=loss_fn)
        test_loss, test_acc = test_step(model=model,
                                        device=device,
                                        test_data=test_data,
                                        loss_fn=loss_fn)
        print(f"epoch: {epoch+1} train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}, test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}")