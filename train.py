import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from data_augmentation import mixup_cutmix_data


def get_sgd_optimizer(model, lr, momentum=0.9, weight_decay=5e-4, nesterov=False):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov
    )
    return optimizer


def get_lr_scheduler(optimizer, num_epochs, steps_per_epoch, learning_rate):
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        # total_steps=total_steps, #provide either total_steps or (epochs and steps_per_epoch)
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        # pct_start=0.3,  # 30% of training for warmup
        pct_start=0.25, # reach peak LR by 25% of total steps
        anneal_strategy='cos',
        # div_factor=10.0,  # initial_lr = max_lr/10
        # final_div_factor=100.0  # min_lr = max_lr/100
        div_factor=25,  # start_lr = 0.2 / 25 = 0.008
        final_div_factor=1e4  # end_lr = 0.2 / 10_000 = 2e-5
    )
    #Should print ~0.008 (for div_factor=25).
    print("Initial LR:", optimizer.param_groups[0]['lr'])
    return scheduler


def get_cosine_scheduler(optimizer, max_lr, num_epochs, steps_per_epoch, warmup_epochs=2):
    # Fine-tuning scheduler: short warmup + cosine decay

    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    # # manually set optimizer LR start (low warmup start) -- not needed, start_factor will take care of this
    # for g in optimizer.param_groups:
    #     g['lr'] = max_lr / 10

    warmup = LinearLR(
        optimizer,
        start_factor=1/10, # start at 10% of base LR
        total_iters=warmup_steps # number of scheduler.step() calls during warmup
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps, # number of remaining updates
        eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs] # switch to cosine after warmup
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
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    best_loss = checkpoint['best_val_loss']

    if use_amp and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"✅ Checkpoint loaded. Resuming from epoch {start_epoch}")

    return start_epoch, best_loss


def print_diagnostics(pbar, model, scaler, batch_idx, use_amp):
    """
    Print gradient diagnostics:
      - Total & max grad norms
      - NaN / Inf / Zero gradient checks
      - GradScaler info (if AMP enabled)
    """
    total_norm_sq = 0.0
    max_norm = 0.0
    grad_none_count = 0
    nan_layers, inf_layers, zero_layers = [], [], []

    #Print Gradient norm for debugging. If grad norm ≈ 0 for many updates, learning is not happening.
    # If inf/nan, there's numerical instability.
    print("\n=== Gradient norms per layer (first few layers) ===")
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_data = p.grad.data
            grad_norm = grad_data.norm(2).item()
            total_norm_sq += grad_norm ** 2
            max_norm = max(max_norm, grad_norm)

            # Detect anomalies
            if torch.isnan(grad_data).any():
                nan_layers.append(name)
            elif torch.isinf(grad_data).any():
                inf_layers.append(name)
            elif torch.allclose(grad_data, torch.zeros_like(grad_data)):
                zero_layers.append(name)
            # count += 1
            # if count <= 10:  # limit printout
            #     print(f"{name:<40s} grad_norm={grad_norm:.6e}")
        else:
            grad_none_count += 1
            # Only print first few missing grads to avoid spam
            if grad_none_count <= 5:
                print(f"{name:<40s} grad=None")

    total_norm = total_norm_sq ** 0.5
    pbar.write(f"[Grad Debug] Step {batch_idx}: total_norm={total_norm:.6e}, max_norm={max_norm:.6e}")

    # Print anomalies (if any)
    if nan_layers:
        pbar.write(f"[Warning] NaN gradients in: {', '.join(nan_layers[:5])}{'...' if len(nan_layers) > 5 else ''}")
    if inf_layers:
        pbar.write(f"[Warning] Inf gradients in: {', '.join(inf_layers[:5])}{'...' if len(inf_layers) > 5 else ''}")
    if zero_layers:
        pbar.write(f"[Info] Zero gradients in: {', '.join(zero_layers[:5])}{'...' if len(zero_layers) > 5 else ''}")

    if not (nan_layers or inf_layers or zero_layers):
        pbar.write("[Grad Debug] No NaN/Inf/Zero gradients detected ✅")

    # AMP / GradScaler info
    if use_amp and scaler is not None:
        scale_val = scaler.get_scale()
        pbar.write(f"[Grad Debug] GradScaler scale={scale_val:.1f}")
    # else:
    #     pbar.write(f"[Grad Debug] AMP disabled — GradScaler inactive.")


def update_ema(model, ema_model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            # ema_p.copy_(ema_p * decay + p * (1 - decay))
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


def train_loop(model, device, train_loader, optimizer, scheduler, scaler, train_losses, train_acc,
               accumulation_steps=4, use_amp=True, debug_every=200, ema_model=None, ema_decay=0.9999):
    """
    Training loop for one epoch with gradient accumulation, mixed precision, OneCycleLR per batch, and LR logging
    """
    model.train()
    pbar = tqdm(train_loader, desc="Training", leave=False)
    correct = 0
    processed = 0
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    # On some GPUs (A100, H100, etc.) FP16 underflows. Use torch.bfloat16 instead if supported
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    current_lr = optimizer.param_groups[0]["lr"]

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        # Apply MixUp or CutMix
        data, targets_a, targets_b, lam = mixup_cutmix_data(
            data, target, alpha=0.2, cutmix_prob=0.5,
            use_cutmix=True, use_mixup=True
        )

        # 🧠 Check input stats for zero / NaN inputs. Expected: mean ≈ 0, std ≈ 1, min/max roughly around -2.1 and +2.6
        if batch_idx == 0:  # print for only first batch to avoid spam
            pbar.write(f"[Debug] Input mean={data.mean().item():.4f}, std={data.std().item():.4f}, "
                  f"min={data.min().item():.4f}, max={data.max().item():.4f}")
            pbar.write(f"[Debug] Device check: model={next(model.parameters()).device}, data={data.device}")
            pbar.write(f"[Debug] Batch shape: {tuple(data.shape)}")
            pbar.write(f"[Debug] Label range: {target.min().item()}–{target.max().item()}")

        # Forward + loss under autocast
        with autocast(enabled=use_amp, dtype=dtype):
            y_pred = model(data)
            # ✅Compute smoothed loss for both targets (MixUp / CutMix)
            loss = lam * F.cross_entropy(y_pred, targets_a, label_smoothing=0.1) \
                   + (1 - lam) * F.cross_entropy(y_pred, targets_b, label_smoothing=0.1)
            loss = loss / accumulation_steps

        # Track unscaled loss (for logging)
        train_losses.append(loss.detach().item() * accumulation_steps)

        # ✅ Backward pass (AMP vs non-AMP)
        if use_amp: # scale the loss
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation step - Update weights only after accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
            if batch_idx < 20: #log first N batch losses
                pbar.write(f"step {global_step} LR={optimizer.param_groups[0]['lr']:.6f} batch_loss={loss.item() * accumulation_steps:.4f}")
            global_step += 1

            # 🧩 2️⃣ Gradient norm diagnostic every N steps
            if batch_idx % debug_every == 0:
                print_diagnostics(pbar, model, scaler, batch_idx, use_amp)
                pbar.write(f"[MixAug Debug] lam={lam:.3f} | "
                           f"targets_a[0]={targets_a[0].item()} | targets_b[0]={targets_b[0].item()}")

            if use_amp:
                scaler.unscale_(optimizer)

            # Add gradient clipping to prevent instability in the first few thousand steps.
            # clip_grad_norm clips the gradients in place, and returns the total gradient norm before clipping
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if batch_idx % debug_every == 0:
                pbar.write(f"[Grad Debug] Before step: total_norm={total_norm:.2e}")

            # ✅ Optimizer step (AMP vs non-AMP)
            if use_amp:
                scaler.step(optimizer) # Step optimizer through scaler
                scaler.update()
                scale_val = scaler.get_scale()
                if scale_val < 1:
                    pbar.write(f"[Warning] GradScaler scale dropped below 1 — possible inf/NaN gradients.")
            else:
                optimizer.step()

            # Update EMA model
            update_ema(model, ema_model, ema_decay)

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
            val_loss += F.cross_entropy(output, target, reduction='sum', label_smoothing=0.1).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    val_loss /= len(val_loader.dataset)  # per-sample average loss
    acc = 100.0 * correct / len(val_loader.dataset)

    val_losses.append(val_loss)
    val_acc.append(acc)

    print(f"\nVal set: Avg loss: {val_loss:.4f}, "
          f"Accuracy: {correct}/{len(val_loader.dataset)} ({acc:.2f}%)\n")

    return val_losses, val_acc