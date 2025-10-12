import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from torchvision import datasets, transforms as tf

class InspectImage:

    def __init__(self, dataset, words_file=None):
        """
        Initialize with a PyTorch ImageFolder dataset and optional words.txt file.
        
        Args:
            dataset: PyTorch dataset (e.g., ImageFolder)
            words_file: Path to words.txt to map WordNet IDs to readable names
        """
        self.dataset = dataset
        self.wnid_to_name = self.load_wnid_classnames(words_file) if words_file else None

    def load_wnid_classnames(self, words_file):
        """
        Loads WordNet ID to readable class name mapping from words.txt
        """
        wnid_to_name = {}
        if not os.path.exists(words_file):
            print(f"Warning: words.txt file not found at {words_file}. Using raw IDs.")
            return wnid_to_name

        with open(words_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                wnid = parts[0]
                classname = parts[1].split(",")[0]  # pick first readable name
                wnid_to_name[wnid] = classname
        return wnid_to_name

    def inspect_loader(self, loader_name="Loader"):
        """
        Prints statistics about the dataset
        """
        dataset = self.dataset
        print(f"\n===== {loader_name} =====")
        
        # Total number of images
        total_images = len(dataset)
        print(f"Total images: {total_images}")
        
        # List of class labels
        class_labels = dataset.classes
        print(f"Class labels: {class_labels[:10]} ... (showing first 10)")
        
        # Total number of classes
        num_classes = len(class_labels)
        print(f"Total number of classes: {num_classes}")
        
        # Number of images per class
        class_counts = Counter([label for _, label in dataset.samples])
        print("Number of images per class (first 10):")
        for cls_idx in range(min(10, num_classes)):
            cls_name = class_labels[cls_idx]
            display_name = self.wnid_to_name.get(cls_name, cls_name) if self.wnid_to_name else cls_name
            print(f"  {display_name}: {class_counts[cls_idx]}")
        
        # Image size (check first image)
        first_img, _ = dataset[0]
        print(f"Image size: {first_img.size()}")  # torch.Size([C, H, W])

    def show_images_per_class(self, num_classes=10, images_per_class=3, loader_name="Loader"):
        """
        Displays images with readable class names
        """
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
            np_img = img.numpy().transpose((1, 2, 0))  # C,H,W -> H,W,C
            plt.imshow(np_img)
            plt.axis('off')
            
            display_name = self.wnid_to_name.get(cls_name, cls_name) if self.wnid_to_name else cls_name
            plt.title(display_name, fontsize=8)

            if all(v == images_per_class for v in class_img_count.values()):
                break

        plt.tight_layout()
        plt.show()