import os
import time
import datetime
import zipfile
import torch
import torch.optim as optim
import gc
from data_loader import generate_train_val_loader, generate_hf_train_val_loader
from model import ResNet50
from train import train_loop, val_loop, get_sgd_optimizer, get_lr_scheduler, get_cosine_scheduler, load_checkpoint, save_checkpoint
from utils import InspectImage
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import copy

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


def create_ema_model(model):
    # Create a copy of the model and update it with EMA weights
    ema_decay = 0.9999
    ema_model = copy.deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model, ema_decay


def main(data_path="./content/tiny-imagenet-200", 
         zip_path="./content/tiny-imagenet.zip",
         batch_size=128, 
         num_epochs=2,
         learning_rate=0.1,
         inspect_data=False,
         checkpoints_dir="./content/drive/MyDrive/checkpoints/resnet50",
         resume_training=False,
         num_workers=8,
         use_amp=True,
         hf_dataset=True,
         experiment_name="MyTrainRun",
         resume_weights_file="best.pth",
         finetuning_run=False
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
    if not hf_dataset:
        if not os.path.exists(data_path):
            print(f"Dataset not found. Attempting to extract from {zip_path}")
            extract_to = os.path.dirname(data_path) or "/content/"
            if not unzip_tiny_imagenet(zip_path, extract_to):
                print("❌ Failed to extract dataset. Exiting...")
                return
        else:
            print(f"✓ Dataset found at: {data_path}")

    # Create output folders if needed
    checkpoints_dir = os.path.join(checkpoints_dir, experiment_name)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir, exist_ok=True)

    # ====== STEP 2: Load Data ======
    print(f"\n[STEP 2/6] Loading dataset and creating data loaders...")
    print(f"  - Batch size: {batch_size}, num_workers: {num_workers}")
    # train_loader, val_loader = generate_train_val_loader(data_path, batch_size=batch_size,train_transform=True, val_transform=True)
    train_loader, val_loader = generate_hf_train_val_loader(batch_size=batch_size, train_transform=True, val_transform=True, num_workers=num_workers)
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
    # model = ResNet50(num_classes=200, use_maxpool=False)
    model = ResNet50(num_classes=1000, use_maxpool=True)
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
    if finetuning_run:
        weight_decay = 1e-4
        nesterov = True
    else:
        weight_decay = 5e-4
        nesterov = False
    optimizer = get_sgd_optimizer(model, learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=nesterov)

    accumulation_steps = 4
    steps_per_epoch = len(train_loader) // accumulation_steps
    print(f"✓ Optimizer: SGD (lr={learning_rate}, momentum=0.9, weight_decay={weight_decay}), nesterov={nesterov}")
    print(f"  - Max LR: {learning_rate}")
    print(f"  - Total steps: {steps_per_epoch * num_epochs}")

    if finetuning_run:
        # Learning Rate Strategy while finetuning: warmup + cosine annealing
        # For finetuning run, use learning_rate=0.01. Should see val accuracy jump earlier and rise past 70% within ~10 epochs.
        warmup_epochs = num_epochs // 10
        scheduler = get_cosine_scheduler(optimizer, learning_rate, num_epochs, steps_per_epoch, warmup_epochs)
        print(f"✓ LR Scheduler: CosineAnnealingLR with warmup of {warmup_epochs} epochs")
    else:
        # Learning Rate Strategy while training from scratch: OneCycleLR
        # OneCycleLR expects total_steps = num_epochs * steps_per_epoch. Since we are dividing loss by accumulation_steps, we are
        # effectively making the model step fewer times. So LR schedule runs too fast relative to optimization steps and needs to be handled
        scheduler = get_lr_scheduler(optimizer, num_epochs, steps_per_epoch, learning_rate)

        # --- Diagnostic: check warmup length ---
        print(f"✓ LR Scheduler: OneCycleLR")
        total_steps = getattr(scheduler, "total_steps", num_epochs * steps_per_epoch)
        pct_start = getattr(scheduler, "pct_start", 0.08)  # default to your chosen warmup %
        warmup_steps = int(total_steps * pct_start)
        print(f"Total steps: {total_steps:,}, Warmup steps: {warmup_steps:,} ({pct_start*100:.1f}% of training)")

    # ====== STEP 6: Training Loop ======
    print(f"\n[STEP 6/6] Starting training...")
    print("="*70)

    # Tracking metrics
    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []
    best_loss = float('inf')
    best_weights_file = os.path.join(checkpoints_dir, resume_weights_file)

    # Create Scaler if mixed precision is true
    scaler = GradScaler(enabled=use_amp)  # handles scaling automatically

    # Tensorboard writer
    writer = SummaryWriter(f'/Data/tf_runs/{experiment_name}')  # or simply SummaryWriter()

    if resume_training and os.path.exists(best_weights_file):
        # Resume from best weights, or from last epoch?
        start_epoch, best_loss = load_checkpoint(model, optimizer, scaler, best_weights_file, device, use_amp)

        # Fast-forward scheduler so it resumes from the correct LR step
        completed_steps = start_epoch * steps_per_epoch
        for _ in range(completed_steps):
            scheduler.step()
        print(f"[Resume] Scheduler fast-forwarded to epoch {start_epoch}, step {completed_steps}.")
        print(f"[Resume] Current LR = {optimizer.param_groups[0]['lr']:.6f}")
    else:
        start_epoch = 1

    # ✅ Create EMA *after* loading weights
    ema_model, ema_decay = create_ema_model(model)

    for epoch in range(start_epoch, num_epochs + 1):
        start_time = time.time()
        print(f"\n{'='*70}")
        print(f"📊 EPOCH {epoch}/{num_epochs}")
        print(f"{'='*70}")
        
        # Training
        print("\n🔄 Training...")
        train_losses, train_acc = train_loop(model, device, train_loader, optimizer, scheduler, scaler, train_losses, train_acc,
                                             accumulation_steps=accumulation_steps, use_amp=use_amp,
                                             ema_model=ema_model, ema_decay=ema_decay)
        
        # Validation
        print("\n🔍 Validating...")
        val_losses, val_acc = val_loop(
            ema_model, device, val_loader, val_losses, val_acc, use_amp=use_amp
        )
        
        # Print epoch summary
        print(f"\n📈 Epoch {epoch} Summary:")
        print(f"  - Train Loss: {train_losses[-1]:.4f}")
        print(f"  - Train Acc: {train_acc[-1]:.2f}%")
        print(f"  - Val Loss: {val_losses[-1]:.4f}")
        print(f"  - Val Acc: {val_acc[-1]:.2f}%")
        print(f"  - Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model if validation loss improved. Also save every x epochs?
        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            print(f"Validation loss improved to {best_loss:.4f}. Saving model weights to {best_weights_file}")
            save_checkpoint(model, optimizer, scaler, epoch, best_loss, val_losses[-1], best_weights_file, use_amp)


        # Save every epoch as well, for backup
        epoch_weights_file = os.path.join(checkpoints_dir, f"epoch-{epoch}.pth")
        print(f"Saving epoch weights: {epoch_weights_file}")
        save_checkpoint(model, optimizer, scaler, epoch, best_loss, val_losses[-1], epoch_weights_file, use_amp)
        torch.save(ema_model.state_dict(), f"epoch-{epoch}-ema.pth")

        # Aggregate epoch metrics
        train_loss_epoch = sum(train_losses[-len(train_loader):]) / len(train_loader)
        train_acc_epoch = train_acc[-1]
        val_loss_epoch = sum(val_losses[-len(val_loader):]) / len(val_loader)
        val_acc_epoch = val_acc[-1]

        # Log to Tensorboard
        writer.add_scalar("Loss/train", train_loss_epoch, epoch)
        writer.add_scalar("Accuracy/train", train_acc_epoch, epoch)
        writer.add_scalar("Loss/val", val_loss_epoch, epoch)
        writer.add_scalar("Accuracy/val", val_acc_epoch, epoch)

        # Log learning rate
        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f"LR/group_{i}", param_group["lr"], epoch)

        secs = time.time() - start_time
        print(f"Time taken for epoch {epoch}: {str(datetime.timedelta(seconds = secs))}")

    # ====== Final Summary ======
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  - Best Train Accuracy: {max(train_acc):.2f}%")
    print(f"  - Best Val Accuracy: {max(val_acc):.2f}%")
    print(f"  - Final Train Loss: {train_losses[-1]:.4f}")
    print(f"  - Final Val Loss: {val_losses[-1]:.4f}")
    print("="*70)

    writer.close()

    return model, train_losses, train_acc, val_losses, val_acc


if __name__ == "__main__":

    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()

    # Set memory fraction (use only 80% of GPU memory)
    torch.cuda.set_per_process_memory_fraction(0.7)

   
    # Enable memory efficient settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Add this to check GPU memory
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Allocated memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
        print(f"Reserved memory: {torch.cuda.memory_reserved() / 1e9:.1f} GB")    

    # For Onecycle Policy:
    # base_lr ≈ 0.1 * (batch_size / 256)

    # Configuration for Colab testing
    # Adjust these parameters based on your needs

    # # For Colab testing (mounted Google Drive)
    # model, *metrics = main(
    #     data_path="./content/tiny-imagenet-200",
    #     zip_path="./content/tiny-imagenet.zip",
    #     batch_size=8,  # Increase if you have enough GPU memory
    #     num_epochs=2,    # Test with 1-2 epochs
    #     learning_rate=0.1,
    #     inspect_data=False,  # Set True to see dataset stats
    #     checkpoints_dir="./content/drive/MyDrive/checkpoints/resnet50",
    #     num_workers=8,
    #     use_amp=True,
    #     hf_dataset=True,
    #     # experiment_name=""
    # )

    # # For g5.2xlarge. Fresh run after fixes -- with aMP
    # model, *metrics = main(
    #     data_path="",
    #     zip_path="",
    #     batch_size=352, #368,#384 # Increase if you have enough GPU memory
    #     num_epochs=90,
    #     learning_rate=0.1,
    #     inspect_data=False,  # Set True to see dataset stats
    #     checkpoints_dir="/Data/checkpoints",
    #     num_workers=16,
    #     use_amp=True,
    #     hf_dataset=True,
    #     experiment_name="Run3-lr-fixes",
    #     resume_training=False,
    # )

    # # Without AMP - smaller batch size, change lr
    # # For g5.2xlarge. Fresh run after fixes
    # model, *metrics = main(
    #     data_path="",
    #     zip_path="",
    #     batch_size=176, #352, #368,#384 # Increase if you have enough GPU memory
    #     num_epochs=90,
    #     learning_rate=0.05,
    #     inspect_data=False,  # Set True to see dataset stats
    #     checkpoints_dir="/Data/checkpoints",
    #     num_workers=16,
    #     use_amp=False,
    #     hf_dataset=True,
    #     experiment_name="Run4-lr-fixes",
    #     resume_training=False,
    # )


    # For g5.2xlarge. Resuming previous run from epoch67 -- with aMP
    # Keep LR and scheduler params same, batch size can be changed
    model, *metrics = main(
        data_path="",
        zip_path="",
        batch_size=352, #368,#384 # Increase if you have enough GPU memory
        num_epochs=90,
        learning_rate=0.05,
        inspect_data=False,  # Set True to see dataset stats
        checkpoints_dir="/Data/checkpoints",
        num_workers=16,
        use_amp=True,
        hf_dataset=True,
        experiment_name="Run5-amp-epoch67",
        resume_training=True,
        resume_weights_file="run4-nonAMP_epoch66.pth"
    )
