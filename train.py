import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


def get_sgd_optimizer():
    pass

def get_lr_scheduler(optimizer, num_epochs, steps_per_epoch, learning_rate):

    total_steps = steps_per_epoch * num_epochs

    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,  # 30% of training for warmup
        anneal_strategy='cos',
        div_factor=10.0,  # initial_lr = max_lr/10
        final_div_factor=100.0  # min_lr = max_lr/100
    )
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
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    best_loss = checkpoint['best_val_loss']

    if use_amp and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"✅ Checkpoint loaded. Resuming from epoch {start_epoch}")

    return start_epoch, best_loss


def train_loop(model, device, train_loader, optimizer, scaler, train_losses, train_acc, accumulation_steps=4):
    """
    Training loop for one epoch with gradient accumulation and mixed precision option
    """
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Forward + loss under autocast
        with autocast(dtype=torch.float16): # or bfloat16 on newer GPUs (e.g. A100, H100)
            # Predict
            y_pred = model(data)

            # Calculate loss (divide by accumulation steps)
            #loss = F.nll_loss(y_pred, target) / accumulation_steps
            loss = F.cross_entropy(y_pred, target) / accumulation_steps

        # Track unscaled loss (for logging)
        train_losses.append(loss.detach().item() * accumulation_steps)

        # Scale the loss and perform backward pass
        scaler.scale(loss).backward()

        # Update weights only after accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Step optimizer through scaler
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f'Loss={loss.item() * accumulation_steps} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
        train_acc.append(100 * correct / processed)

    return train_losses, train_acc


def val_loop(model, device, val_loader, val_losses, val_acc, use_amp):
    """
    Val loop for one epoch with mixed precision option
    """
    model.eval()
    val_loss = 0
    correct = 0
    # with torch.no_grad(), autocast(dtype=torch.float16):  # same precision context
    with torch.no_grad(), autocast(enabled=use_amp):
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # val_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            val_loss += F.cross_entropy(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    val_acc.append(100. * correct / len(val_loader.dataset))

    return val_losses, val_acc
