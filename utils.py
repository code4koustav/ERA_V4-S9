import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from torchvision import datasets, transforms as tf
import torch

class InspectImage:

    def __init__(self, dataset, words_file=None, normalize=False, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
        """
        Initialize with a PyTorch ImageFolder dataset or AlbumentationsImageDataset.
        """
        self.dataset = getattr(dataset, "dataset", dataset)  # support Albumentations wrapper
        self.wnid_to_name = self.load_wnid_classnames(words_file) if words_file else None
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def load_wnid_classnames(self, words_file):
        wnid_to_name = {}
        if not os.path.exists(words_file):
            print(f"Warning: words.txt file not found at {words_file}. Using raw IDs.")
            return wnid_to_name
        with open(words_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                wnid = parts[0]
                classname = parts[1].split(",")[0]
                wnid_to_name[wnid] = classname
        return wnid_to_name

    def inspect_loader(self, loader_name="Loader"):
        dataset = self.dataset
        print(f"\n===== {loader_name} =====")
        total_images = len(dataset)
        print(f"Total images: {total_images}")
        class_labels = dataset.classes
        print(f"Class labels: {class_labels[:10]} ... (showing first 10)")
        num_classes = len(class_labels)
        print(f"Total number of classes: {num_classes}")
        class_counts = Counter([label for _, label in dataset.samples])
        print("Number of images per class (first 10):")
        for cls_idx in range(min(10, num_classes)):
            cls_name = class_labels[cls_idx]
            display_name = self.wnid_to_name.get(cls_name, cls_name) if self.wnid_to_name else cls_name
            print(f"  {display_name}: {class_counts[cls_idx]}")
        first_img, _ = dataset[0]
        if isinstance(first_img, torch.Tensor):
            print(f"Image size: {first_img.size()}")
        else:
            print(f"Image size: {np.array(first_img).shape}")

    def show_images_per_class(self, num_classes=10, images_per_class=3, loader_name="Loader"):
        dataset = self.dataset
        class_labels = dataset.classes
        class_img_count = {cls: 0 for cls in class_labels[:num_classes]}
        plt.figure(figsize=(images_per_class * 3, num_classes * 3))
        count = 0

        print(f'############# Images for {loader_name} ####################')
        for img, label in dataset:
            cls_name = class_labels[label]
            if cls_name not in class_img_count:
                continue
            if class_img_count[cls_name] >= images_per_class:
                continue

            class_img_count[cls_name] += 1
            count += 1

            plt.subplot(num_classes, images_per_class, count)
            if isinstance(img, torch.Tensor):
                np_img = img.numpy().transpose((1, 2, 0))
            else:
                np_img = np.array(img)

            if self.normalize:
                np_img = (np_img * np.array(self.std) + np.array(self.mean))
                np_img = np.clip(np_img, 0, 1)

            plt.imshow(np_img)
            plt.axis('off')
            display_name = self.wnid_to_name.get(cls_name, cls_name) if self.wnid_to_name else cls_name
            plt.title(display_name, fontsize=8)

            if all(v == images_per_class for v in class_img_count.values()):
                break

        plt.tight_layout()
        plt.show()

    # -----------------------------
    # New method: visualize augmentations
    # -----------------------------
    def show_augmented_images(self, transform, num_images=5, samples_per_image=3):
        """
        Visualize original vs augmented images for Albumentations transforms.

        Args:
            transform: Albumentations transform
            num_images: number of images to pick from dataset
            samples_per_image: number of random augmentations per image
        """
        for i in range(num_images):
            img, label = self.dataset[i]
            if isinstance(img, torch.Tensor):
                img_np = img.numpy().transpose(1,2,0)
            else:
                img_np = np.array(img)

            plt.figure(figsize=(4 * (samples_per_image + 1), 4))

            # Show original
            plt.subplot(1, samples_per_image + 1, 1)
            plt.imshow(img_np)
            plt.title("Original")
            plt.axis('off')

            # Show augmented versions
            for j in range(samples_per_image):
                augmented = transform(image=img_np)["image"]  # <-- use img_np here
                aug_np = augmented.permute(1,2,0).numpy() if isinstance(augmented, torch.Tensor) else np.array(augmented)
                if self.normalize:
                    aug_np = (aug_np * np.array(self.std) + np.array(self.mean))
                    aug_np = np.clip(aug_np, 0, 1)
                plt.subplot(1, samples_per_image + 1, j + 2)
                plt.imshow(aug_np)
                plt.title(f"Augmented {j+1}")
                plt.axis('off')

            plt.tight_layout()
            plt.show()