from .models import from_type
from .training import train, get_class_weights
from .datasets import get_dataloaders, CityscapesDataset
import torch
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import os
import time

OUTPUT_DIR = "models/"
MODEL_TYPE = "segformer"

if __name__ == "__main__":
    # use slurm job id or timestamp
    JOB_ID = os.environ.get("SLURM_JOB_ID", "local_" + str(int(time.time())))

    dtype = torch.float32  

    if torch.cuda.is_available():
        device = torch.device("cuda") # For NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    else:
        device = torch.device("cpu")

    model, _dataset = from_type(MODEL_TYPE)
    train_loader, val_loader = get_dataloaders(CityscapesDataset, batch_size=32, num_workers=4)

    print("Starting training on device: ", device)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    class_weights = get_class_weights(
        train_loader, num_classes=19, device=device, dtype=dtype
    )
    print("Class weights computed. Starting training...")
    losses = train(
        model,
        optimizer,
        train_loader,
        val_loader,
        25,
        device,
        dtype,
        class_weights,
        scheduler=scheduler,
    )

    DIR = OUTPUT_DIR + JOB_ID + "/"

    os.makedirs(DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{DIR}{MODEL_TYPE}.pth")

    image = Image.open(
        "data/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png"
    ).convert("RGB")
        
    transform = transforms.Compose(
        [transforms.Resize((256, 512)), transforms.ToTensor()]
    )

    image = transform(image)
    image = image.to(device) # type: ignore
    model = model.to(device=device)
    mask = model(image.unsqueeze(0))
    mask = mask.cpu().detach().numpy().squeeze(0)
    print(mask[:, 0, 0])

    plt.savefig(f"{DIR}example_output.png")

    # Plot loss curves and save
    plt.figure()
    plt.plot(losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.savefig(f"{DIR}loss_curves.png")
    plt.close()


