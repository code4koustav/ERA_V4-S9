"""
Utilities to load ImageNet-style datasets.
Reorganizes Tiny ImageNet val split and builds Albumentations datasets/DataLoaders.
"""

import torch
from torch.utils.data import DataLoader
from data_augmentation import AlbumentationsImageDataset, HFDatasetWrapper, get_train_transform, get_test_transform
import os
import shutil
from datasets import load_dataset

def reorganize_val_folder(val_dir):
    """
    Convert Tiny ImageNet val folder to ImageFolder layout.
    Moves images into class subfolders using val_annotations.txt; no-op if organized.
    """
    img_dir = os.path.join(val_dir, "images")
    ann_file = os.path.join(val_dir, "val_annotations.txt")

    if not os.path.exists(img_dir):
        return

    with open(ann_file) as f:
        for line in f:
            img_name, class_id = line.strip().split('\t')[:2]
            class_dir = os.path.join(val_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)
            shutil.move(os.path.join(img_dir, img_name), os.path.join(class_dir, img_name))

    shutil.rmtree(img_dir)
    print("✅ Validation folder reorganized successfully!")


def load_imagenet_dataset(data_path):
    """
    Create train/val Albumentations datasets from an ImageNet-style root.
    Auto-fixes Tiny ImageNet val layout and returns (train_dataset, val_dataset).
    """
    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")
    reorganize_val_folder(val_dir)

    train_dataset = AlbumentationsImageDataset(train_dir, transform=get_train_transform())
    val_dataset = AlbumentationsImageDataset(val_dir, transform=get_test_transform())
    return train_dataset, val_dataset


def generate_train_val_loader(data_path, batch_size=64, train_transform=True, test_transform=True):
    """
    Creates DataLoader objects for training and validation sets.
    
    Args:
        data_path (str): Path to ImageNet/Tiny ImageNet folder
        batch_size (int): Batch size
        train_transform (bool): Apply training augmentations if True
        test_transform (bool): Apply test/validation transformations if True
    """

    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")

    # Optional: reorganize validation folder for Tiny ImageNet
    from data_loader import reorganize_val_folder
    reorganize_val_folder(val_dir)

    # Select transforms based on flags
    train_tf = get_train_transform() if train_transform else None
    val_tf = get_test_transform() if test_transform else None

    # Create datasets
    train_dataset = AlbumentationsImageDataset(train_dir, transform=train_tf)
    val_dataset = AlbumentationsImageDataset(val_dir, transform=val_tf)

    # CUDA settings
    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    if cuda:
        torch.cuda.manual_seed(1)

    dataloader_args = dict(
        batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True
    ) if cuda else dict(batch_size=batch_size, shuffle=True)

    train_loader = DataLoader(train_dataset, **dataloader_args)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=False
    )

    return train_loader, val_loader


def generate_hf_train_val_loader(batch_size=64, train_transform=True, test_transform=True, num_workers=8):
    """
    Creates DataLoader objects for training and validation sets from Huggingface.

    Args:
        batch_size (int): Batch size
        train_transform (bool): Apply training augmentations if True
        test_transform (bool): Apply test/validation transformations if True
    """

    # Load dataset from huggingface cache dir
    train_dataset = load_dataset(
        "ILSVRC/imagenet-1k",
        split="train",
        cache_dir="/Data/datasets_cache"
    )

    val_dataset = load_dataset(
        "ILSVRC/imagenet-1k",
        split="validation",
        cache_dir="/Data/datasets_cache"
    )

    # Select transforms based on flags
    train_tf = get_train_transform() if train_transform else None
    val_tf = get_test_transform() if test_transform else None

    train_dataset = HFDatasetWrapper(train_dataset, transform=train_tf)
    val_dataset = HFDatasetWrapper(val_dataset, transform=val_tf)

    # CUDA settings
    cuda = torch.cuda.is_available()
    torch.manual_seed(1)

    dataloader_args = dict(
        batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True
    ) if cuda else dict(batch_size=batch_size, shuffle=True)

    train_loader = DataLoader(train_dataset, **dataloader_args)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader