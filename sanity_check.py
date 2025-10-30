import torch
import torch.nn.functional as F
from data_loader import generate_hf_train_val_loader
from model import ResNet50
import matplotlib.pyplot as plt
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR


def one_batch_sanity_check(model, train_loader, device):
    """
    What to expect:
    cross_entropy should be ~6–8 at initialization for ImageNet (1000 classes)
    Calling nll_loss on raw logits will likely error or give nonsense.
    If cross_entropy is tiny (~0.02) but accuracy is tiny → you have a label mismatch or your outputs are already
    transformed wrongly (e.g., model returned probabilities and you applied log twice).
    """

    # one-batch sanity check
    model.eval()
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    with torch.no_grad():
        outputs = model(data)    # should be logits (no softmax/log_softmax)
    print("outputs shape:", outputs.shape)  # (B, num_classes)
    print("outputs dtype:", outputs.dtype)
    print("target dtype:", target.dtype, "min/max:", target.min().item(), target.max().item())

    # Compute cross-entropy and NLL to compare
    ce = F.cross_entropy(outputs, target, reduction='mean').item()
    nll_try = None
    try:
        nll_try = F.nll_loss(outputs, target, reduction='mean').item()
    except Exception as e:
        nll_try = f"error: {e}"
    print("cross_entropy:", ce, "   nll_loss_on_logits:", nll_try)


def batch_data_check(train_loader):
    # Check small category distribution
    ys = []
    for i, (_, t) in enumerate(train_loader):
        ys.append(t)
        if i >= 10: break
    ys = torch.cat(ys)
    print("Unique labels (sample):", torch.unique(ys)[:20])
    print("Label dtype:", ys.dtype)


def batch_overfit_check(model, train_loader, device):
    # Overfit 1 batch. This proves the model & optimizer can fit data.
    # Expected: loss → near 0, accuracy → 100% after some iterations. If not, there is a model/optimizer bug
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        single_x, single_y = x[:8].clone(), y[:8].clone()  # small mini "dataset"
        break

    for it in range(200):
        optimizer.zero_grad()
        out = model(single_x)
        loss = F.cross_entropy(out, single_y)
        loss.backward()
        optimizer.step()
        if it % 20 == 0:
            pred = out.argmax(1)
            acc = (pred == single_y).float().mean().item() * 100
            print(f"it {it} loss {loss.item():.4f} acc {acc:.2f}%")


def check_lr_scheduler(model, train_loader, epochs, max_lr):
    accumulation_steps = 4
    steps_per_epoch = len(train_loader)
    effective_steps_per_epoch = steps_per_epoch // accumulation_steps

    # total_steps = steps_per_epoch * num_epochs

    optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.9)

    # Define OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=effective_steps_per_epoch,
        pct_start=0.3,
        div_factor=10, #25,
        final_div_factor=1e4,
        anneal_strategy='cos'
    )

    print("Initial lr:", optimizer.param_groups[0]['lr'])
    # simulate a few scheduler steps (if you have scheduler)
    for step in range(20):
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        print(step, optimizer.param_groups[0]['lr'])


def validate_loss_computation(model, val_loader, device):
    # Minimal reliable val loss
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            batch_loss = F.cross_entropy(out, y, reduction='sum').item()
            total_loss += batch_loss
            total += y.size(0)
    avg = total_loss / total
    print("Val avg loss (per-sample):", avg)


def visualize_scheduler(epochs, batch_size, max_lr):
    accumulation_steps = 4
    #steps_per_epoch = len(train_loader) // accumulation_steps
    steps_per_epoch = 1000  # Example number of batches before accumulation adjustment
    effective_steps_per_epoch = steps_per_epoch // accumulation_steps
    # max_lr = 0.2

    # Dummy model and optimizer
    model = torch.nn.Linear(10, 2)
    optimizer = SGD(model.parameters(), lr=max_lr)

    # Define OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=effective_steps_per_epoch,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1e4,
        anneal_strategy='cos'
    )

    # Simulate LR steps
    lrs = []
    for _ in range(epochs * effective_steps_per_epoch):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    # Plot LR schedule
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.title("OneCycleLR Schedule (max_lr=0.2, 90 epochs, accumulation=4, batch=352)")
    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    # plt.show()
    plt.savefig("OneCycleLR-plot.png")
    plt.close()


def check_batch_sanity(batch_size, num_workers, model=None, train_loader=None):
    epochs = 90
    max_lr = 0.2

    if not train_loader:
        train_loader, val_loader = generate_hf_train_val_loader(batch_size=batch_size, train_transform=True,
                                                                val_transform=True, num_workers=num_workers)

    if not model:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNet50(num_classes=1000, use_maxpool=True)
        model = model.to(device)

    print(f"\n\n=== 1. Confirm model outputs and loss usage")
    one_batch_sanity_check(model, train_loader, device)

    print(f"\n\n=== 2. Verify labels and dataset sizes")
    batch_data_check(train_loader)

    print(f"\n\n=== 3. Overfit a single batch test, Check if loss is reducing correctly")
    batch_overfit_check(model, train_loader, device)

    print(f"\n\n=== 4. Print current lr and scheduler behavior after a few steps")
    check_lr_scheduler(model, train_loader, epochs, max_lr)

    print(f"\n\n=== 5. Validate val_loss computation")
    validate_loss_computation(model, val_loader, device)

    print(f"\n\n=== 6. Visualize Scheduler (saved in  OneCycleLR-plot.png")
    visualize_scheduler(epochs, batch_size, max_lr)

if __name__ == "__main__":
    check_batch_sanity(batch_size=352, num_workers=18)
