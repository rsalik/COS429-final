from .training import train, get_class_weights
from .datasets import get_dataloaders
import torch
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=32, num_workers=4)

    dtype = torch.float32  # We will be using float throughout.

    if torch.cuda.is_available():
        device = torch.device("cuda")  # For NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    else:
        device = torch.device("cpu")

    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        classes=19,
        activation=None,
    )

    print(device)
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    class_weights = get_class_weights(
        train_loader, num_classes=19, device=device, dtype=dtype
    )
    train(model, optimizer, train_loader, val_loader, 5, device, dtype, class_weights)

    torch.save(model.state_dict(), "unet.pth")

    image = Image.open(
        "data/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png"
    ).convert("RGB")
    plt.imshow(image)

    transform = transforms.Compose(
        [transforms.Resize((256, 512)), transforms.ToTensor()]
    )

    image = transform(image)
    image = image.to(device)
    model = model.to(device=device)
    mask = model(image.unsqueeze(0))
    mask = mask.cpu().detach().numpy().squeeze(0)
    print(mask[:, 0, 0])
    plt.imshow(mask.argmax(0), cmap="tab20")
