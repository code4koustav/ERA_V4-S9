import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
import torch
from torch.utils.data import Dataset
import numpy as np

# -----------------------------
# Albumentations-based transformations
# -----------------------------
def get_train_transform():
    return A.Compose([
        # Randomly crop a region of the image and resize it to 224x224.
        # Simulates zooming in or out while preserving aspect ratio.
        # scale=(0.08, 1.0): crop between 8% to 100% of original area
        # ratio=(3/4, 4/3): aspect ratio distortion to mimic various object shapes
        A.RandomResizedCrop(size=(224,224), scale=(0.08, 1.0), ratio=(3/4, 4/3), p=1.0),

        # Randomly flip the image horizontally with a 50% chance
        # Helps model generalize for left/right symmetry in objects (e.g., animals, vehicles)
        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(shift_limit=0.0625,  # ~6% translation
            scale_limit=0.1,  # zoom in/out up to ±10%
            rotate_limit=15,  # rotate ±15°
            border_mode=0,  # reflect padding
            p=0.5
        ),

        # Randomly apply either ColorJitter or HueSaturationValue (not both)
        # Mimics lighting, color tone, and saturation variations
        A.OneOf([
            # Randomly change brightness, contrast, saturation, and hue
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            # Randomly shift hue, saturation, and value in HSV space
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
        ], p=0.8),  # 80% chance to apply one of the color transformations

        # Randomly convert image to grayscale with 20% chance
        # Forces model to learn shape/structure rather than relying only on color
        A.OneOf([
            A.ToGray(p=1.0),
            A.NoOp()
        ], p=0.2),

        # Randomly blur the image to simulate out-of-focus or motion blur
        # blur_limit=(3, 5): kernel size between 3x3 and 5x5
        A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 2.0), p=0.3),

        # CoarseDropout (a.k.a Random Erasing or Cutout)
        # Randomly removes rectangular patches from the image to make model robust
        # Encourages learning from surrounding context rather than specific pixels
        A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            fill=(0, 0, 0),
            fill_mask=None
        ),

        # Normalize using ImageNet mean and std
        # Keeps input distribution consistent with pretrained models and stabilizes training
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        # Convert image and label to PyTorch tensors
        ToTensorV2()
    ])


def get_val_transform():
    return A.Compose([
        # Resize smaller edge to 256 and preserve aspect ratio
        A.Resize(height=256, width=256),

        # Crop the center 224x224 region — standard ImageNet evaluation crop
        A.CenterCrop(height=224, width=224),

        # Normalize with ImageNet mean and std (same as training)
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        # Convert to tensor for model input
        ToTensorV2()
    ])


def mixup_cutmix_data(x, y, alpha=0.2, cutmix_prob=0.5, use_cutmix=True, use_mixup=True):
    """Apply MixUp or CutMix to a batch"""
    if not use_cutmix and not use_mixup:
        return x, y, y, 1.0  # unchanged

    lam = 1.0
    rand_index = torch.randperm(x.size(0)).to(x.device)

    if use_cutmix and np.random.rand() < cutmix_prob:
        # --- CutMix ---
        lam = np.random.beta(alpha, alpha)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    elif use_mixup:
        # --- MixUp ---
        lam = np.random.beta(alpha, alpha)
        x = lam * x + (1 - lam) * x[rand_index, :]

    y_a, y_b = y, y[rand_index]
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    """Helper: generate random bounding box for CutMix"""
    W, H = size[3], size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# -----------------------------
# Dataset wrapper for Albumentations
# -----------------------------
class AlbumentationsImageDataset(Dataset):
    """
    Wraps torchvision ImageFolder dataset so that Albumentations transforms can be applied.
    """
    def __init__(self, image_folder, transform=None):
        self.dataset = datasets.ImageFolder(image_folder)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # image = np.array(image)  # Convert PIL -> NumPy
        image = np.array(image, dtype=np.uint8)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

# -----------------------------
# Dataset wrapper for Huggingface Dataset
# -----------------------------

class HFDatasetWrapper(Dataset):
    """
    Wraps a Hugging Face image dataset for use with Albumentations transforms.
    """
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # print(f"In getitem of HFDatasetWrapper: {idx}")
        sample = self.dataset[idx]
        image = sample["image"]

        # Ensure 3 channels (convert grayscale → RGB)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # image = np.array(image) # PIL -> NumPy
        image = np.array(image, dtype=np.uint8)
        label = sample["label"]

        # print("Type:", type(image), "Shape:", getattr(image, "shape", None))
        if self.transform:
            image = self.transform(image=image)["image"]

        # # Ensure it's a torch.Tensor
        # if not torch.is_tensor(image):
        #     # image = torch.from_numpy(image).permute(2, 0, 1)
        #     image = torch.tensor(image)
        # image = image.contiguous()

        return image, torch.tensor(label)