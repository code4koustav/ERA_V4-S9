import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from data_augmentation import mixup_cutmix_data, get_late_stage_transforms
from monitor import get_system_stats, log_ema_diff, get_post_clip_gradnorm, print_diagnostics


def get_sgd_optimizer(model, lr, momentum=0.9, weight_decay=5e-4, nesterov=False):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov
    )
    return optimizer


def get_adam_optimizer(model, lr):
    #base_lr = lr
    #backbone_lr = base_lr * 0.5  # half of that for pretrained layers

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
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


def get_cosine_scheduler(optimizer, max_lr, num_epochs, steps_per_epoch, warmup_epochs=2, start_factor=0.01):
    # Fine-tuning scheduler: short warmup + cosine decay

    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    tmax = total_steps - warmup_steps
    eta_min = 1e-6

    # # manually set optimizer LR start (low warmup start) -- not needed, start_factor will take care of this
    # for g in optimizer.param_groups:
    #     g['lr'] = max_lr / 10

    warmup = LinearLR(
        optimizer,
        start_factor=start_factor, # start at 1% of base LR
        total_iters=warmup_steps # number of scheduler.step() calls during warmup
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=tmax, # number of remaining updates
        eta_min=eta_min
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps] # switch to cosine after warmup
    )
    return scheduler, tmax, warmup_steps, eta_min


def save_checkpoint(model, ema_model, optimizer, scheduler, scaler, epoch, best_loss, epoch_val_loss,
                    path, use_amp, lr_scheduler_type, lr_scheduler_params):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "ema_model_state": ema_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        'best_val_loss': best_loss,
        'epoch_val_loss': epoch_val_loss,
        "lr_scheduler_type": lr_scheduler_type,
        "lr_scheduler_params": lr_scheduler_params,
        # Add any other relevant information like hyperparameters
    }

    if hasattr(scheduler, "state_dict"):
        checkpoint["scheduler_state"] = scheduler.state_dict()

    # Save AMP scaler state only if AMP is enabled
    if use_amp:
        checkpoint["scaler_state"] = scaler.state_dict()

    torch.save(checkpoint, path)
    print(f"✅ Checkpoint saved to {path}")


def load_checkpoint(model, ema_model, optimizer, scheduler, scaler, path, device, use_amp,
                    current_lr_type, current_lr_params, steps_per_epoch):
    """
    Load checkpoint safely. Detect LR scheduler mismatch to avoid optimizer/scheduler state errors.
    Returns: start_epoch, best_loss
    """
    checkpoint = torch.load(path, map_location=device)

    # Load model + EMA model weights
    model.load_state_dict(checkpoint["model_state"])
    if "ema_model_state" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_model_state"])
    else:
        # First time: initialize EMA from main model
        ema_model.load_state_dict(model.state_dict())

    # --------------------------
    # Detect LR scheduler mismatch
    # --------------------------
    checkpoint_lr_type = checkpoint.get("lr_scheduler_type")
    checkpoint_lr_params = checkpoint.get("lr_scheduler_params", {})
    lr_mismatch = (checkpoint_lr_type != current_lr_type) or (checkpoint_lr_params != current_lr_params)
    if lr_mismatch:
        print(f"[Resume] LR scheduler mismatch detected: checkpoint={checkpoint_lr_type}, current={current_lr_type}")
        print(f"[Resume] Loaded {path} weights, Skipping optimizer/scaler/scheduler state loading.")
        load_opt_scaler_scheduler = False
    else:
        load_opt_scaler_scheduler = True

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_loss = checkpoint.get('best_val_loss', float('inf'))

    # --------------------------
    # Load optimizer/scaler/scheduler if safe
    # --------------------------
    if load_opt_scaler_scheduler:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if use_amp and "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
            print(f"[Resume] Scaler state restored from checkpoint for AMP.")

        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            print(f"[Resume] Scheduler state restored from checkpoint.")
        else:
            # Fast-forward scheduler to current epoch so it resumes from the correct LR step
            completed_steps = start_epoch * steps_per_epoch
            for _ in range(completed_steps):
                scheduler.step()
            print(f"[Resume] Scheduler fast-forwarded to epoch {start_epoch}, step {completed_steps}.")

        print(f"[Resume] Current LR = {optimizer.param_groups[0]['lr']:.6f}")


    print(f"✅ Checkpoint loaded. Resuming from epoch {start_epoch}")

    return start_epoch, best_loss


def maybe_switch_to_fine_tune_phase(epoch, optimizer, train_loader, switch_epoch=20, min_lr=2e-5, pbar=None):
    """
    Dynamically switch to fine-tune phase:
      - lighter augmentations
      - lower LR
      (EMA intentionally disabled for stability)
    """

    if epoch >= switch_epoch:
        print(f"[Phase Switch] Epoch {epoch}: switching to light augmentations & lower LR")
        if pbar:
            pbar.write(f"[Phase Switch] Epoch {epoch}: switching to light augmentations & lower LR")

        # --- Update augmentations ---
        # if hasattr(train_loader.dataset, "transform"):
        dataset = getattr(train_loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "transform"):
            # train_loader.dataset.transform = get_late_stage_transforms()
            dataset.transform = get_late_stage_transforms()
            print("✓ Updated training augmentations for fine-tuning phase.")
            if pbar:
                pbar.write("✓ Updated training augmentations for fine-tuning phase.")

        # --- Reduce LR by 2× and floor it at min_lr ---
        # Only do this if switch_epoch is not too low. otherwise run is too short
        if switch_epoch >= 15:
            for param_group in optimizer.param_groups:
                old_lr = param_group["lr"]
                new_lr = max(old_lr * 0.5, min_lr)
                param_group["lr"] = new_lr
            print(f"✓ Reduced LR to {optimizer.param_groups[0]['lr']:.6f}")
            if pbar:
                pbar.write(f"✓ Reduced LR to {optimizer.param_groups[0]['lr']:.6f}")


def update_ema(model, ema_model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            # ema_p.copy_(ema_p * decay + p * (1 - decay))
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


def train_loop(model, device, train_loader, optimizer, scheduler, scaler, train_losses, train_acc, epoch,
               accumulation_steps=4, use_amp=True, ema_model=None, ema_decay=0.9999,
               current_cutmix_prob=0.5, current_mixup_prob=0.5, logger=None, label_smoothing=0.1):
    """
    Training loop for one epoch with gradient accumulation, mixed precision, OneCycleLR per batch, and LR logging
    """
    model.train()
    pbar = tqdm(train_loader, desc="Training", leave=False)
    correct = 0
    processed = 0
    global_step = 0
    gpu_utils = []
    optimizer.zero_grad(set_to_none=True)
    pre_clip_norm = 0
    post_clip_norm = 0

    # On some GPUs (A100, H100, etc.) FP16 underflows. Use torch.bfloat16 instead if supported
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    current_lr = optimizer.param_groups[0]["lr"]
    mixup_active = (current_mixup_prob > 0.0) or (current_cutmix_prob > 0.0)
    label_smoothing_when_mix = 0.0  # don't additionally smooth when using mixup

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        # Apply MixUp or CutMix
        if mixup_active:
            data, targets_a, targets_b, lam = mixup_cutmix_data(
                data, target, alpha=0.2, cutmix_prob=current_cutmix_prob, mixup_prob=current_mixup_prob,
                use_cutmix=True, use_mixup=True
            )
        else:
            targets_a, targets_b, lam = target, target, 1.0  # fall back to standard mode

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
            if mixup_active and lam < 1.0:
                loss = lam * F.cross_entropy(y_pred, targets_a, label_smoothing=label_smoothing_when_mix) \
                       + (1 - lam) * F.cross_entropy(y_pred, targets_b, label_smoothing=label_smoothing_when_mix)
            else:
                loss = F.cross_entropy(y_pred, target, label_smoothing=label_smoothing)

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

            if use_amp:
                scaler.unscale_(optimizer)

            # Add gradient clipping to prevent instability in the first few thousand steps.
            # clip_grad_norm clips the gradients in place, and returns the total gradient norm before clipping
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            post_clip_norm = get_post_clip_gradnorm(model)

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

            # Check EMA update. If EMA is working, this difference should start small and gradually increase, reflecting smoothing.
            if batch_idx < 20 or batch_idx % 200 == 0:  # log every 100 batches
                log_ema_diff(model, ema_model, step=batch_idx, pbar=None)
                print_diagnostics(pbar, model, scaler, batch_idx, use_amp, ema_model)


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

        # --- Accuracy tracking (mix-aware) ---

        stats = get_system_stats()  # CPU, RAM, GPU
        if logger:
            logger.log(
                epoch=epoch,
                batch_idx=batch_idx,
                batch_loss=loss.item() * accumulation_steps,
                batch_acc=acc,
                lr=current_lr,
                grad_norm=post_clip_norm,
                gpu_util=stats["gpu"],
                cpu_util=stats["cpu"],
                ram_util=stats["ram"],
                gpu_mem=stats["gpu_mem"],
            )

        # Update tqdm
        gpu_util = stats["gpu"] if stats["gpu"] is not None else "N/A"
        gpu_mem = stats["gpu_mem"] if stats["gpu_mem"] is not None else "N/A"
        pbar.set_description(
            f"Loss={loss.item() * accumulation_steps:.4f} | "
            f"Batch={batch_idx} | Acc={acc:.2f}% | LR={current_lr:.6f} | "
            f"GradNorm(Pre/Post)={pre_clip_norm:.4f}/{post_clip_norm:.4f} | GPU Util={gpu_util} | GPU mem={gpu_mem}"
        )

    if gpu_utils:
        avg_util = sum(gpu_utils) / len(gpu_utils)
        print(f"✅ Avg GPU utilization this epoch: {avg_util:.1f}%")
        
    return train_losses, train_acc


def val_loop(model, device, val_loader, val_losses, val_acc, use_amp, label_smoothing=0.1):
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
            val_loss += F.cross_entropy(output, target, reduction='sum', label_smoothing=label_smoothing).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    val_loss /= len(val_loader.dataset)  # per-sample average loss
    acc = 100.0 * correct / len(val_loader.dataset)

    val_losses.append(val_loss)
    val_acc.append(acc)

    print(f"\nVal set: Avg loss: {val_loss:.4f}, "
          f"Accuracy: {correct}/{len(val_loader.dataset)} ({acc:.2f}%)\n")

    return val_losses, val_acc
