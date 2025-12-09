from .utils import accuracy
import torch


def train(
    model,
    optimizer,
    loader_train,
    loader_val,
    epochs,
    device,
    dtype,
    class_weights,
    scheduler=None,
    print_every=10,
):
    """
    Train a model using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - loader_train: A dataloader containing the train dataset
    - loader_val: A dataloader containing the validation dataset
    - epochs: (Optional) An integer giving the number of epochs to train for
    - print_every: (Optional) An integer specifying how often to print the loss.
    - scheduler: (Optional) A PyTorch learning rate scheduler.

    Returns: Nothing, but prints model losses and accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    losses = []
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = torch.nn.functional.cross_entropy(
                scores, y.squeeze(1), ignore_index=255, weight=class_weights
            )
            losses.append(loss.item())

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print("Epoch {}, iteration {}, loss = {}".format(e, t, loss.item()))

        print("Epoch {} done".format(e))
        accuracy(model, loader_val, device, dtype)

        if scheduler is not None:
            scheduler.step()

    return losses
