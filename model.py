import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet50
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet architecture
    """
    def __init__(self, block, layers, num_classes=1000, in_channels=3, use_maxpool=True):
        """
        Args:
            block: Block type (BasicBlock or Bottleneck)
            layers: List of number of blocks in each layer
            num_classes: Number of output classes (default 1000 for ImageNet, use 200 for Tiny ImageNet)
            in_channels: Number of input channels (default 3 for RGB images)
            use_maxpool: Whether to use maxpool after initial conv (True for ImageNet 224x224, False for Tiny ImageNet 64x64)
        """
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.use_maxpool = use_maxpool
        
        # Initial convolution layer
        # For full ImageNet (224x224): 7x7 kernel, stride 2
        # For Tiny ImageNet (64x64): 3x3 kernel, stride 1 (set via parameters)
        if use_maxpool:
            # Standard ImageNet configuration
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, 
                                   stride=2, padding=3, bias=False)
        else:
            # Tiny ImageNet configuration
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, 
                                   stride=1, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Maxpool for ImageNet (224x224), skip for Tiny ImageNet (64x64)
        if use_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Create a ResNet layer with multiple blocks
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """
        Initialize weights using He initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        if self.use_maxpool:
            x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        # Return without log probablities - using CrossEntropyLoss loss for mixed precision
        # return F.log_softmax(x, dim=1)
        return x  # raw logits (no log_softmax)

def ResNet50(num_classes=1000, in_channels=3, use_maxpool=True):
    """
    ResNet50 model
    
    Args:
        num_classes: Number of output classes 
                     - 1000 for full ImageNet (default)
                     - 200 for Tiny ImageNet
        in_channels: Number of input channels (default 3 for RGB)
        use_maxpool: Whether to use maxpool after initial conv
                     - True for ImageNet 224x224 (default)
                     - False for Tiny ImageNet 64x64
    
    Returns:
        ResNet50 model instance
    
    Examples:
        # For full ImageNet (224x224, 1000 classes)
        model = ResNet50(num_classes=1000, use_maxpool=True)
        
        # For Tiny ImageNet (64x64, 200 classes)
        model = ResNet50(num_classes=200, use_maxpool=False)
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, 
                  in_channels=in_channels, use_maxpool=use_maxpool)

