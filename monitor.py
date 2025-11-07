import psutil
import pynvml
import torch
import time
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import csv, os
from datetime import datetime

# Initialize NVIDIA Management Library
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0 by default

def get_system_stats():
    """Return dict of current CPU, RAM, and GPU utilization."""
    cpu_util = psutil.cpu_percent(interval=None)
    ram_util = psutil.virtual_memory().percent

    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_mem_used = gpu_mem_info.used / gpu_mem_info.total * 100

    return {
        "cpu": cpu_util,
        "ram": ram_util,
        "gpu": gpu_util,
        "gpu_mem": gpu_mem_used,
    }


# -----------------------------
# Training logger class
# -----------------------------
class TrainLogger:
    def __init__(self, log_dir="logs", experiment_name="train_run", log_to_csv=True, log_to_stdout=True):
        self.log_to_csv = log_to_csv
        self.log_to_stdout = log_to_stdout
        self.log_path = None

        if self.log_to_csv:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = os.path.join(log_dir, f"{experiment_name}_{timestamp}.csv")

            # Write header
            with open(self.log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "batch_idx", "batch_loss", "batch_acc",
                    "lr", "grad_norm", "gpu_util", "cpu_util", "ram_util", "gpu_mem"
                ])

    def log(self, epoch, batch_idx, batch_loss, batch_acc, lr,
            grad_norm=None, gpu_util=None, cpu_util=None, ram_util=None, gpu_mem=None):

        if self.log_to_csv and self.log_path:
            with open(self.log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, batch_idx, batch_loss, batch_acc,
                    lr,
                    grad_norm if grad_norm is not None else "",
                    gpu_util if gpu_util is not None else "",
                    cpu_util if cpu_util is not None else "",
                    ram_util if ram_util is not None else "",
                    gpu_mem if gpu_mem is not None else "",

                ])

        if self.log_to_stdout:
            # log every x batches
            if batch_idx % 100 == 0:
                msg = (f"Epoch:{epoch} Batch:{batch_idx} | "
                       f"Loss:{batch_loss:.4f} Acc:{batch_acc:.4f} "
                       f"LR:{lr:.2e} GradNorm:{grad_norm if grad_norm is not None else 'N/A'} "
                       f"GPU:{gpu_util if gpu_util is not None else 'N/A'} "
                       f"CPU:{cpu_util if cpu_util is not None else 'N/A'} "
                       f"RAM:{ram_util if ram_util is not None else 'N/A'}")
                print(msg)

    def info(self):
        return f"Logging to {self.log_path}"



def get_post_clip_gradnorm(model):
    # Compute post-clip norm for logging
    post_clip_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            post_clip_norm += param_norm.item() ** 2
    post_clip_norm = post_clip_norm ** 0.5
    return post_clip_norm


def print_diagnostics(pbar, model, scaler, batch_idx, use_amp, ema_model=None):
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

    # # Print system stats
    # stats = get_system_stats()
    # pbar.write(f"[Batch {batch_idx:05d}] CPU: {stats['cpu']:5.1f}% | RAM: {stats['ram']:5.1f}% | "
    #       f"GPU: {stats['gpu']:5.1f}% | GPU-Mem: {stats['gpu_mem']:5.1f}%")


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
    print(f"[Grad Debug] Step {batch_idx}: total_norm={total_norm:.6e}, max_norm={max_norm:.6e}")

    # Print anomalies (if any)
    if nan_layers:
        print(f"[Warning] NaN gradients in: {', '.join(nan_layers[:5])}{'...' if len(nan_layers) > 5 else ''}")
    if inf_layers:
        print(f"[Warning] Inf gradients in: {', '.join(inf_layers[:5])}{'...' if len(inf_layers) > 5 else ''}")
    if zero_layers:
        print(f"[Info] Zero gradients in: {', '.join(zero_layers[:5])}{'...' if len(zero_layers) > 5 else ''}")

    if not (nan_layers or inf_layers or zero_layers):
        print("[Grad Debug] No NaN/Inf/Zero gradients detected ✅")

    # AMP / GradScaler info
    if use_amp and scaler is not None:
        scale_val = scaler.get_scale()
        print(f"[Grad Debug] GradScaler scale={scale_val:.1f}")
    # else:
    #     pbar.write(f"[Grad Debug] AMP disabled — GradScaler inactive.")


def visualize_augmentations(dataset, n_images=16, nrow=4, save_path=None):
    """
    Visualize a few augmented samples from a dataset to sanity-check transforms.

    Args:
        dataset: torch.utils.data.Dataset (e.g., your training dataset)
        n_images: total number of images to show
        nrow: grid columns
        save_path: if given, saves the plot instead of showing it
    """
    # temporarily disable shuffling if DataLoader wraps dataset
    idxs = np.random.choice(len(dataset), size=n_images, replace=False)

    imgs = []
    for i in idxs:
        img, label = dataset[i]
        if isinstance(img, torch.Tensor):
            imgs.append(img)
        else:
            # some transforms may return numpy arrays
            imgs.append(torch.from_numpy(np.array(img)).permute(2, 0, 1))

    grid = torchvision.utils.make_grid(imgs, nrow=nrow, normalize=True, scale_each=True)
    npimg = grid.cpu().numpy().transpose((1, 2, 0))

    plt.figure(figsize=(nrow * 2.5, n_images / nrow * 2.5))
    plt.imshow(np.clip(npimg, 0, 1))
    plt.axis("off")
    plt.title("Sample Augmented Images", fontsize=14)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"✅ Saved augmentation preview to {save_path}")
    else:
        plt.show()


def measure_dataloader_speed(dataloader, num_batches=100):
    """
    Measure average time (in seconds) spent per batch in the DataLoader.
    Only measures data loading, not forward/backward pass.
    """
    start_time = time.time()
    n_batches = 0

    data_iter = iter(dataloader)
    for _ in range(num_batches):
        try:
            _ = next(data_iter)
            n_batches += 1
        except StopIteration:
            break

    total_time = time.time() - start_time
    avg_time = total_time / max(1, n_batches)

    print(f"🔍 DataLoader profiling: {n_batches} batches")
    print(f"⏱️  Average batch load time: {avg_time:.4f} sec")
    print(f"⚡ Approx. samples/sec (per worker): {dataloader.batch_size / avg_time / dataloader.num_workers:.1f}")
    return avg_time


def log_ema_diff(model, ema_model, step, pbar=None, topk_layers=3):
    """Logs both absolute and relative EMA differences."""
    if ema_model is None:
        return 0.0, 0.0

    abs_diff, abs_ref = 0.0, 0.0
    layer_diffs = []

    with torch.no_grad():
        for i, (p, q) in enumerate(zip(model.parameters(), ema_model.parameters())):
            d = torch.sum(torch.abs(p - q)).item()
            abs_diff += d
            abs_ref += torch.sum(torch.abs(p)).item()
            if i < topk_layers:
                layer_diffs.append(d)

    rel_diff = abs_diff / (abs_ref + 1e-8)
    msg = (f"[EMA Debug] Step {step} | AbsDiff: {abs_diff:.3f} | "
           f"RelDiff: {rel_diff:.6e} | First {topk_layers} layer diffs: "
           f"{[round(x, 3) for x in layer_diffs]}")

    if pbar:
        pbar.write(msg)
    else:
        print(msg)

    return abs_diff, rel_diff


# def get_gpu_utilization(device=0):
#     """
#     Returns current GPU utilization %.
#     Uses torch.cuda.utilization() if available (PyTorch 2.1+),
#     otherwise queries NVML via pynvml.
#     """
#     util = None
#     try:
#         # PyTorch 2.1+ API (fast)
#         util = torch.cuda.utilization(device)
#     except AttributeError:
#         try:
#             import pynvml
#             pynvml.nvmlInit()
#             handle = pynvml.nvmlDeviceGetHandleByIndex(device)
#             util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
#         except Exception:
#             util = None
#     return util