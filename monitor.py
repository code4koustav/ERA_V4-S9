import psutil
import pynvml
import torch
import time
import matplotlib.pyplot as plt
import torchvision
import numpy as np

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

    # Print system stats
    stats = get_system_stats()
    print(f"[Batch {batch_idx:05d}] CPU: {stats['cpu']:5.1f}% | RAM: {stats['ram']:5.1f}% | "
          f"GPU: {stats['gpu']:5.1f}% | GPU-Mem: {stats['gpu_mem']:5.1f}%")


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
