# Train ImageNet-1k using ResNet-50

## Project Plan

For testing the pipeline download the Tine Imagenet-200 data

1. ✅ Download data manually and prepare the data loader function in py file
2. ⏳ Create data augmentation function in a py file
3. ✅ Develop the model architecture in a py file
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

### `model.py`
ResNet50 implementation with adaptive configuration for both full ImageNet and Tiny ImageNet:

**Architecture Components:**
- **`Bottleneck`**: Bottleneck block with 1×1 → 3×3 → 1×1 convolutions (expansion=4)
- **`ResNet`**: Main ResNet architecture with flexible configuration
- **`ResNet50()`**: Factory function to create ResNet50 models

**Key Features:**
- Supports **Full ImageNet** (224×224, 1000 classes) and **Tiny ImageNet** (64×64, 200 classes)
- Adaptive architecture via `use_maxpool` parameter
- He initialization for optimal weight initialization
- Outputs log probabilities using `log_softmax` for NLL loss compatibility

### `train.py`
Training and testing loops:
- **`train_loop()`**: Single epoch training with progress tracking
- **`test_loop()`**: Model evaluation on validation set
- Uses NLL loss and tracks accuracy metrics

### `Notebook.ipynb`
Jupyter notebook containing:
- Data extraction and setup process for Tiny ImageNet-200
- Data loader testing and validation
- Dataset inspection and visualization examples
- Interactive exploration of train/validation datasets
- Shows dataset statistics: 100,000 training images, 10,000 validation images across 200 classes

## Dataset Information

### Tiny ImageNet-200 (For Testing)
- **Dataset**: Tiny ImageNet-200
- **Training Images**: 100,000 (500 per class)
- **Validation Images**: 10,000 (50 per class)
- **Classes**: 200
- **Image Size**: 64×64×3 (RGB)
- **Format**: Organized in ImageFolder structure for PyTorch compatibility

### Full ImageNet-1K (Production)
- **Dataset**: ImageNet ILSVRC 2012
- **Training Images**: ~1.28 million
- **Validation Images**: 50,000
- **Classes**: 1000
- **Image Size**: 224×224×3 (RGB)
- **Format**: ImageFolder structure

## ResNet50 Model Architecture

### Model Specifications

**For Full ImageNet (224×224, 1000 classes):**
- **Total Parameters**: 25,557,032 (~25.5M)
- **Initial Conv**: 7×7 kernel, stride 2, 64 filters
- **MaxPool**: 3×3, stride 2
- **Layer 1**: 3 Bottleneck blocks (64 channels)
- **Layer 2**: 4 Bottleneck blocks (128 channels)
- **Layer 3**: 6 Bottleneck blocks (256 channels)
- **Layer 4**: 3 Bottleneck blocks (512 channels)
- **Output**: Global Average Pooling + FC (2048 → 1000)

**For Tiny ImageNet (64×64, 200 classes):**
- **Total Parameters**: 23,910,152 (~23.9M)
- **Initial Conv**: 3×3 kernel, stride 1, 64 filters
- **No MaxPool**: Preserves spatial resolution for smaller images
- **Layers**: Same structure as above
- **Output**: Global Average Pooling + FC (2048 → 200)

### Bottleneck Block Architecture
```
Input (C_in)
  ↓
Conv 1×1 (C_in → C_out) + BatchNorm + ReLU
  ↓
Conv 3×3 (C_out → C_out) + BatchNorm + ReLU
  ↓
Conv 1×1 (C_out → C_out × 4) + BatchNorm
  ↓
Add Residual Connection
  ↓
ReLU → Output (C_out × 4)
```

### Training Configuration
- **Loss Function**: NLL Loss (Negative Log-Likelihood)
- **Output Activation**: Log Softmax
- **Weight Initialization**: He initialization
- **Device Support**: CPU and CUDA GPU

## Usage Examples

### 1. Basic Model Creation

**For Full ImageNet (Production):**
```python
from model import ResNet50
import torch

# Create model for full ImageNet
model = ResNet50(num_classes=1000, use_maxpool=True)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Model info
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
# Output: Total parameters: 25,557,032
```

**For Tiny ImageNet (Testing):**
```python
from model import ResNet50
import torch

# Create model for Tiny ImageNet
model = ResNet50(num_classes=200, use_maxpool=False)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Model info
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
# Output: Total parameters: 23,910,152
```

### 2. Complete Training Setup

```python
import torch
import torch.optim as optim
from model import ResNet50
from data_loader import generate_train_val_loader
from train import train_loop, test_loop

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data (for Tiny ImageNet testing)
data_path = "./tiny-imagenet-200"
batch_size = 64
train_loader, val_loader = generate_train_val_loader(data_path, batch_size)

# Create model
model = ResNet50(num_classes=200, use_maxpool=False).to(device)

# Setup optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training tracking
train_losses = []
train_acc = []
test_losses = []
test_acc = []

# Train for one epoch
for epoch in range(1, 11):  # 10 epochs
    print(f"\nEpoch {epoch}:")
    
    # Training
    train_losses, train_acc = train_loop(
        model, device, train_loader, optimizer, train_losses, train_acc
    )
    
    # Testing
    test_losses, test_acc = test_loop(
        model, device, val_loader, test_losses, test_acc
    )
```

### 3. Model Inference

```python
import torch
from model import ResNet50
from torchvision import transforms
from PIL import Image

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50(num_classes=1000, use_maxpool=True).to(device)
model.eval()

# Load and preprocess image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

image = Image.open("path/to/image.jpg")
input_tensor = transform(image).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    # output is log probabilities
    probabilities = torch.exp(output)
    predicted_class = output.argmax(dim=1).item()

print(f"Predicted class: {predicted_class}")
```

### 4. Save and Load Model

```python
import torch
from model import ResNet50

# Save model
model = ResNet50(num_classes=1000, use_maxpool=True)
torch.save(model.state_dict(), 'resnet50_imagenet.pth')

# Load model
model = ResNet50(num_classes=1000, use_maxpool=True)
model.load_state_dict(torch.load('resnet50_imagenet.pth'))
model.eval()
```

## Model Parameters

| Parameter | Full ImageNet | Tiny ImageNet |
|-----------|--------------|---------------|
| `num_classes` | 1000 | 200 |
| `use_maxpool` | True | False |
| `in_channels` | 3 (RGB) | 3 (RGB) |
| Input Size | 224×224 | 64×64 |
| Total Params | 25.5M | 23.9M |
