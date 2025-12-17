from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import os

from torch.utils.data import DataLoader
from torchvision import transforms

class CityscapesUNetDataset(Dataset):
    def __init__(self, tvt, image_transform=None, mask_transform=None):
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.images = []
        self.masks = []
        self.tvt = tvt  # 0 for train, 1 for val, 2 for test
        if tvt == 0:
            s = "train"
        elif tvt == 1:
            s = "val"
        else:
            s = "test"

        img_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "leftImg8bit",
            s,
        )
        mask_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "gtFine",
            s,
        )

        for city in os.listdir(img_root):
            img_dir = os.path.join(img_root, city)
            mask_dir = os.path.join(mask_root, city)

            for file_name in os.listdir(img_dir):
                if file_name.endswith("_leftImg8bit.png"):
                    img_path = os.path.join(img_dir, file_name)
                    mask_name = file_name.replace(
                        "_leftImg8bit.png", "_gtFine_labelIds.png"
                    )
                    mask_path = os.path.join(mask_dir, mask_name)

                    if os.path.isfile(img_path) and os.path.isfile(mask_path):
                        self.images.append(img_path)
                        self.masks.append(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, id):
        image = Image.open(self.images[id]).convert("RGB")
        mask = Image.open(self.masks[id])

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, mask
    

class CityscapesSegFormerDataset(Dataset):
    def __init__(self, tvt, image_transform=None, mask_transform=None):
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.images = []
        self.masks = []

        if tvt == 0:
            split = "train"
        elif tvt == 1:
            split = "val"
        else:
            split = "test"

        img_root = os.path.join("leftImg8bit", split)
        mask_root = os.path.join("gtFine_trainvaltest/gtFine", split)

        for city in os.listdir(img_root):
            img_dir = os.path.join(img_root, city)
            mask_dir = os.path.join(mask_root, city)

            for fname in os.listdir(img_dir):
                if fname.endswith("_leftImg8bit.png"):
                    img_path = os.path.join(img_dir, fname)
                    mask_name = fname.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
                    mask_path = os.path.join(mask_dir, mask_name)

                    self.images.append(img_path)
                    self.masks.append(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask_np = mask.squeeze().numpy() # type: ignore


        return image, mask

image_transform = transforms.Compose(
    [
        transforms.Resize(
            (256, 512), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # precomputed mean, std for imgnet. no need to recompute each time
    ]
)

mapping = {
    0: 19,
    1: 19,
    2: 19,
    3: 19,
    4: 19,
    5: 19,
    6: 19,
    7: 0,
    8: 1,
    9: 19,
    10: 19,
    11: 2,
    12: 3,
    13: 4,
    14: 19,
    15: 19,
    16: 19,
    17: 5,
    18: 19,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: 19,
    30: 19,
    31: 16,
    32: 17,
    33: 18,
    -1: 19,
}


def convert(c):
    return mapping[c]


def remap_mask(mask):
    return torch.as_tensor(np.vectorize(convert)(mask), dtype=torch.long)


def mask_transform(mask):
    mask = transforms.functional.resize( # type: ignore
        mask, (256, 512), interpolation=transforms.functional.InterpolationMode.NEAREST # type: ignore
    ) 
    mask = remap_mask(mask)
    return mask


def get_dataloaders(dataset, batch_size=16, num_workers=2):
    train_dataset = dataset(
        tvt=0, image_transform=image_transform, mask_transform=mask_transform
    )
    val_dataset = dataset(
        tvt=1, image_transform=image_transform, mask_transform=mask_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
