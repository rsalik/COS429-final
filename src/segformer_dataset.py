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
        mask_root = os.path.join("gtFinetrainvaltest/gtFine", split)

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
            mask = self.mask_transform(mask)  # Tensor (1,H,W)

        # Convert 0-33 → 0-18 + 255 ignore
        mask_np = mask.squeeze().numpy()
        remapped = np.vectorize(remap_mask)(mask_np)
        mask = torch.tensor(remapped, dtype=torch.long)

        return image, mask


from torch.utils.data import DataLoader
from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225]) #imagenet mean and std since i am too lazy to calculate them myself
])

mapping = {
        0: 255,
        1: 255,
        2: 255,
        3: 255,
        4: 255,
        5: 255,
        6: 255,
        7: 0,
        8: 1,
        9: 255,
        10: 255,
        11: 2,
        12: 3,
        13: 4,
        14: 255,
        15: 255,
        16: 255,
        17: 5,
        18: 255,
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
        29: 255,
        30: 255,
        31: 16,
        32: 17,
        33: 18,
        -1: 255
    }

def convert(c):
  return mapping[c]

def remap_mask(mask):
  return torch.as_tensor(np.vectorize(convert)(mask), dtype=torch.long)

def mask_transform(mask):
  mask = transforms.functional.resize(mask, (256, 512), interpolation=transforms.functional.InterpolationMode.NEAREST)
  mask = remap_mask(mask)
  return mask

'''
mask_transform = transforms.Compose([
    transforms.Resize((256,512), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.PILToTensor(),
])
'''

train_dataset = CityscapesSegFormerDataset(tvt=0, image_transform=image_transform, mask_transform=mask_transform)
val_dataset = CityscapesSegFormerDataset(tvt=1,  image_transform=image_transform, mask_transform=mask_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
