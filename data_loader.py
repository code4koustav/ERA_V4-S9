import torch
from torchvision import datasets, transforms as tf
import os
import shutil
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def reorganize_val_folder(val_dir):
    """
    Reorganize Tiny ImageNet validation folder so each class has its own subfolder.
    """
    img_dir = os.path.join(val_dir, "images")
    ann_file = os.path.join(val_dir, "val_annotations.txt")

    if not os.path.exists(img_dir):
        return  # Already reorganized

    with open(ann_file) as f:
        for line in f:
            img_name, class_id = line.strip().split('\t')[:2]
            class_dir = os.path.join(val_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)
            shutil.move(os.path.join(img_dir, img_name), os.path.join(class_dir, img_name))

    # Remove the empty images folder
    shutil.rmtree(img_dir)
    print("Validation folder reorganized!")

def load_imagenet_dataset(data_path):
    """
    Loads Tiny ImageNet / ImageNet dataset using ImageFolder structure
    """
    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")

    # Reorganize validation folder if not already done
    reorganize_val_folder(val_dir)

    # Convert images to tensor (no other transformation)
    transform = tf.ToTensor()
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

    return train_dataset, val_dataset

def generate_train_val_loader(data_path, batch_size=64):
    """
    Creates DataLoader objects for training and validation sets
    """
    train_dataset, val_dataset = load_imagenet_dataset(data_path)

    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    if cuda:
        torch.cuda.manual_seed(1)

    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=batch_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    val_loader = torch.utils.data.DataLoader(val_dataset, **dataloader_args)

    return train_loader, val_loader
