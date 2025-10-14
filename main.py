import os
import zipfile
import torch
import torch.optim as optim

from data_loader import generate_train_val_loader
from model import ResNet50
from train import train_loop, test_loop, get_lr_scheduler
from utils import InspectImage


def unzip_tiny_imagenet(zip_path, extract_to):
    """
    Unzips tiny-imagenet.zip from content folder
    
    Args:
        zip_path: Path to the zip file (e.g., "/content/tiny-imagenet.zip")
        extract_to: Directory to extract to (e.g., "/content/")
    """
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found!")
        return False
    
    # Check if already extracted
    extracted_folder = os.path.join(extract_to, "tiny-imagenet-200")
    if os.path.exists(extracted_folder):
        print(f"✓ Dataset already extracted at: {extracted_folder}")
        return True
    
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("✓ Extraction complete!")
    return True


def main(data_path="/content/tiny-imagenet-200", 
         zip_path="/content/tiny-imagenet.zip",
         batch_size=128, 
         num_epochs=2, 
         learning_rate=0.1,
         inspect_data=False,
         checkpoints_dir="checkpoints",
         resume_training=False
         ):
    """
    Main function to run the complete training pipeline
    
    Args:
        data_path: Path to tiny-imagenet-200 folder
        zip_path: Path to zip file (if needs extraction)
        batch_size: Batch size for training
        num_epochs: Number of epochs to train (1-2 for testing)
        learning_rate: Maximum learning rate for OneCycleLR
        inspect_data: Whether to show dataset inspection
    """
    print("="*70)
    print("🚀 ImageNet Training Pipeline - ResNet50 on Tiny ImageNet")
    print("="*70)
    
    # ====== STEP 1: Unzip Dataset (if needed) ======
    print("\n[STEP 1/6] Checking dataset...")
    if not os.path.exists(data_path):
        print(f"Dataset not found. Attempting to extract from {zip_path}")
        extract_to = os.path.dirname(data_path) or "/content/"
        if not unzip_tiny_imagenet(zip_path, extract_to):
            print("❌ Failed to extract dataset. Exiting...")
            return
    else:
        print(f"✓ Dataset found at: {data_path}")

    # Create output folders if needed
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir, exist_ok=True)

    # ====== STEP 2: Load Data ======
    print(f"\n[STEP 2/6] Loading dataset and creating data loaders...")
    print(f"  - Batch size: {batch_size}")
    train_loader, val_loader = generate_train_val_loader(data_path, batch_size=batch_size,train_transform=True, test_transform=True)
    print(f"✓ Train loader: {len(train_loader.dataset)} images, {len(train_loader)} batches")
    print(f"✓ Val loader: {len(val_loader.dataset)} images, {len(val_loader)} batches")
    
    # ====== STEP 3: Inspect Dataset (Optional) ======
    if inspect_data:
        print("\n[STEP 3/6] Inspecting dataset...")
        words_file = os.path.join(data_path, "words.txt")
        inspector = InspectImage(train_loader.dataset, words_file=words_file)
        inspector.inspect_loader("Training Set")
    else:
        print("\n[STEP 3/6] Skipping dataset inspection (set inspect_data=True to enable)")
    
    # ====== STEP 4: Initialize Model ======
    print(f"\n[STEP 4/6] Initializing ResNet50 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Device: {device}")
    
    # For Tiny ImageNet: 200 classes, 64x64 images, no maxpool
    model = ResNet50(num_classes=200, use_maxpool=False)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created: ResNet50")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # ====== STEP 5: Setup Training (Optimizer + LR Scheduler) ======
    print(f"\n[STEP 5/6] Setting up optimizer and LR scheduler...")
    
    # Optimizer: SGD with momentum
    optimizer = optim.SGD(model.parameters(), 
                         lr=learning_rate, 
                         momentum=0.9, 
                         weight_decay=5e-4)

    # Learning Rate Strategy: OneCycleLR
    steps_per_epoch = len(train_loader)
    scheduler = get_lr_scheduler(optimizer, num_epochs, steps_per_epoch, learning_rate)
    
    print(f"✓ Optimizer: SGD (lr={learning_rate}, momentum=0.9, weight_decay=5e-4)")
    print(f"✓ LR Scheduler: OneCycleLR")
    print(f"  - Max LR: {learning_rate}")
    print(f"  - Total steps: {steps_per_epoch * num_epochs}")
    
    # ====== STEP 6: Training Loop ======
    print(f"\n[STEP 6/6] Starting training...")
    print("="*70)
    
    # Tracking metrics
    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    best_loss = float('inf')
    best_weights_file = os.path.join(checkpoints_dir, 'best.pth')

    if resume_training and os.path.exists(best_weights_file):
        checkpoint = torch.load(best_weights_file, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_val_loss']
        print(f"\nFound previous checkpoint. Resuming training from {start_epoch} epoch(s)...")
    else:
        start_epoch = 1


    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n{'='*70}")
        print(f"📊 EPOCH {epoch}/{num_epochs}")
        print(f"{'='*70}")
        
        # Training
        print("\n🔄 Training...")
        train_losses, train_acc = train_loop(
            model, device, train_loader, optimizer, train_losses, train_acc
        )

        # Step the scheduler after each batch (OneCycleLR steps per batch)
        scheduler.step()
        
        # Validation
        print("\n🔍 Validating...")
        test_losses, test_acc = test_loop(
            model, device, val_loader, test_losses, test_acc
        )
        
        # Print epoch summary
        print(f"\n📈 Epoch {epoch} Summary:")
        print(f"  - Train Loss: {train_losses[-1]:.4f}")
        print(f"  - Train Acc: {train_acc[-1]:.2f}%")
        print(f"  - Val Loss: {test_losses[-1]:.4f}")
        print(f"  - Val Acc: {test_acc[-1]:.2f}%")
        print(f"  - Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model if validation loss improved. Also save every x epochs?
        if test_losses[-1] < best_loss:
            best_loss = test_losses[-1]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_loss,
                # Add any other relevant information like hyperparameters
            }, best_weights_file)
            print(f"Validation loss improved to {best_loss:.4f}. Saving model weights.")


    # ====== Final Summary ======
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  - Best Train Accuracy: {max(train_acc):.2f}%")
    print(f"  - Best Val Accuracy: {max(test_acc):.2f}%")
    print(f"  - Final Train Loss: {train_losses[-1]:.4f}")
    print(f"  - Final Val Loss: {test_losses[-1]:.4f}")
    print("="*70)
    
    return model, train_losses, train_acc, test_losses, test_acc


if __name__ == "__main__":
    # Configuration for Colab testing
    # Adjust these parameters based on your needs
    
    # For local testing (Windows path)
    # model, *metrics = main(
    #     data_path="./tiny-imagenet-200",
    #     zip_path="./content/tiny-imagenet.zip",
    #     batch_size=64,
    #     num_epochs=2,
    #     learning_rate=0.1,
    #     inspect_data=False
    # )
    
    # For Colab testing (mounted Google Drive)
    model, *metrics = main(
        data_path="/content/tiny-imagenet-200",
        zip_path="/content/drive/MyDrive/tiny-imagenet-200.zip",
        batch_size=128,  # Increase if you have enough GPU memory
        num_epochs=2,    # Test with 1-2 epochs
        learning_rate=0.1,
        inspect_data=False,  # Set True to see dataset stats
        checkpoints_dir="/content/drive/MyDrive/checkpoints/resnet50"
    )