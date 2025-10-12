# Train ImageNet-1k using ResNet-50

## Project Plan

For testing the pipeline download the Tine Imagenet-200 data

1. ✅ Download data manually and prepare the data loader function in py file
2. ⏳ Create data augmentation function in a py file
3. ⏳ Develop the model architecture in a py file
4. ⏳ Develop the train and test function along with the suitable LR strategy in a py file
5. ⏳ Create a Main.py file where will be calling the above 4 functions and will test in colab for 1/2 epoch just to make sure our code is running 
6. ⏳ Push our all code to github

## Project Files Description

### `data_loader.py`
Contains the data loading functionality for ImageNet dataset:
- **`reorganize_val_folder()`**: Automatically reorganizes the validation folder structure to match PyTorch's ImageFolder format
- **`load_imagenet_dataset()`**: Loads train and validation datasets with basic tensor conversion
- **`generate_train_val_loader()`**: Creates DataLoader objects with proper configuration for GPU/CPU usage
- Supports automatic validation folder reorganization and GPU acceleration

### `utils.py`
Utility functions for dataset inspection and visualization:
- **`InspectImage` class**: Provides comprehensive dataset analysis capabilities
- **`load_wnid_classnames()`**: Maps WordNet IDs to human-readable class names
- **`inspect_loader()`**: Prints detailed statistics about the dataset (total images, classes, image sizes)
- **`show_images_per_class()`**: Visualizes sample images from each class with readable labels
- Essential for understanding dataset structure and debugging

### `Notebook.ipynb`
Jupyter notebook containing:
- Data extraction and setup process for Tiny ImageNet-200
- Data loader testing and validation
- Dataset inspection and visualization examples
- Interactive exploration of train/validation datasets
- Shows dataset statistics: 100,000 training images, 10,000 validation images across 200 classes

## Dataset Information
- **Dataset**: Tiny ImageNet-200
- **Training Images**: 100,000 (500 per class)
- **Validation Images**: 10,000 (50 per class)
- **Classes**: 200
- **Image Size**: 64x64x3 (RGB)
- **Format**: Organized in ImageFolder structure for PyTorch compatibility
