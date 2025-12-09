import torch


def accuracy(model, dataloader, device, dtype):
    """Compute accuracy of the model on the given dataloader's dataset."""

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)
            scores = model(x)
            _, preds = scores.max(1)
            for i in range(preds.size(0)):
                num_correct += (preds[i] == y[i]).sum().item()
            num_samples += preds.size(0) * preds.size(1) * preds.size(2)

        acc = float(num_correct) / num_samples
        print(
            "[Acc] Got %d / %d correct (%.2f)" % (num_correct, num_samples, 100 * acc)
        )


def get_class_weights(dataloader, num_classes, device, dtype):
    """Compute class weights based on frequency in the dataset."""

    class_counts = torch.zeros(num_classes, device=device, dtype=dtype)
    total_pixels = 0

    for _, y in dataloader:
        y = y.to(device=device, dtype=dtype)
        for c in range(num_classes):
            class_counts[c] += (y == c).sum().item()
        total_pixels += y.numel()

    class_frequencies = class_counts / total_pixels
    class_weights = 1.0 / (class_frequencies + 1e-6)  # +epsilon to avoid divide by zero
    class_weights = class_weights / class_weights.sum() * num_classes
    return class_weights
