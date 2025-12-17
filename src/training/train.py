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

    model = model.to(device=device)
    losses = []
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = torch.nn.functional.cross_entropy(
                scores, y.squeeze(1), ignore_index=255, weight=class_weights
            )
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            # clip for more stable training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            if t % print_every == 0:
                print("Epoch {}, iteration {}, loss = {}".format(e, t, loss.item()))

        print("Epoch {} done".format(e))
        accuracy(model, loader_val, device, dtype)

        if scheduler is not None:
            scheduler.step()

    return losses
