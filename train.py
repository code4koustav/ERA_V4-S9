import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


def get_sgd_optimizer(model, lr, momentum=0.9, weight_decay=5e-4):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def get_lr_scheduler(optimizer, num_epochs, steps_per_epoch, learning_rate):

    # total_steps = steps_per_epoch * num_epochs

    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        # total_steps=total_steps, #provide either total_steps or (epochs and steps_per_epoch)
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,  # 30% of training for warmup
        anneal_strategy='cos',
        # div_factor=10.0,  # initial_lr = max_lr/10
        # final_div_factor=100.0  # min_lr = max_lr/100
        div_factor=25,  # start_lr = 0.2 / 25 = 0.008
        final_div_factor=1e4  # end_lr = 0.2 / 10_000 = 2e-5
    )
    #Should print ~0.008 (for div_factor=25).
    print("Initial LR:", optimizer.param_groups[0]['lr'])
    return scheduler


def save_checkpoint(model, optimizer, scaler, epoch, best_loss, epoch_val_loss, path, use_amp):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        'best_val_loss': best_loss,
        'epoch_val_loss': epoch_val_loss,
        # Add any other relevant information like hyperparameters
    }

    # Save AMP scaler state only if AMP is enabled
    if use_amp:
        checkpoint["scaler_state"] = scaler.state_dict()

    torch.save(checkpoint, path)
    print(f"✅ Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, scaler, path, device, use_amp):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    best_loss = checkpoint['best_val_loss']

    if use_amp and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"✅ Checkpoint loaded. Resuming from epoch {start_epoch}")

    return start_epoch, best_loss


def train_loop(model, device, train_loader, optimizer, scheduler, scaler, train_losses, train_acc,
               accumulation_steps=4, use_amp=True):
    """
    Training loop for one epoch with gradient accumulation, mixed precision, OneCycleLR per batch, and LR logging
    """
    model.train()
    pbar = tqdm(train_loader, desc="Training", leave=False)
    correct = 0
    processed = 0
    optimizer.zero_grad(set_to_none=True)

    # On some GPUs (A100, H100, etc.) FP16 underflows. Use torch.bfloat16 instead if supported
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    current_lr = optimizer.param_groups[0]["lr"]

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        # Forward + loss under autocast
        with autocast(enabled=use_amp, dtype=dtype):
            # Predict
            y_pred = model(data)

            # Calculate loss (divide by accumulation steps)
            loss = F.cross_entropy(y_pred, target) / accumulation_steps

        # Track unscaled loss (for logging)
        train_losses.append(loss.detach().item() * accumulation_steps)

        # Scale the loss and perform backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation step - Update weights only after accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
            # Add gradient clipping to prevent instability in the first few thousand steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Step optimizer through scaler
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # step LR scheduler once per optimizer update (=> after each batch (OneCycleLR steps per batch))
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

        # Accuracy
        pred = y_pred.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        processed += len(data)
        acc = 100.0 * correct / processed
        train_acc.append(acc)

        # Update tqdm
        pbar.set_description(
            f"Loss={loss.item() * accumulation_steps:.4f} | "
            f"Batch={batch_idx} | Acc={acc:.2f}% | LR={current_lr:.6f}"
        )

    return train_losses, train_acc


def val_loop(model, device, val_loader, val_losses, val_acc, use_amp):
    """
    Validation loop for one epoch with mixed precision.
    """
    model.eval()
    val_loss = 0.0
    correct = 0

    # On some GPUs (A100, H100, etc.) FP16 underflows. Use torch.bfloat16 instead if supported
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad(), autocast(enabled=use_amp, dtype=dtype):
        for data, target in tqdm(val_loader, desc="Validating", leave=False):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            val_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    val_loss /= len(val_loader.dataset)  # per-sample average loss
    acc = 100.0 * correct / len(val_loader.dataset)

    val_losses.append(val_loss)
    val_acc.append(acc)

    print(f"\nVal set: Avg loss: {val_loss:.4f}, "
          f"Accuracy: {correct}/{len(val_loader.dataset)} ({acc:.2f}%)\n")

    return val_losses, val_acc