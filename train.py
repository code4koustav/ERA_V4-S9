import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
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


def train_loop(model, device, train_loader, optimizer, train_losses, train_acc, accumulation_steps=4):
    """
    Training loop for one epoch with gradient accumulation
    """
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    optimizer.zero_grad()
    
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Predict
        y_pred = model(data)

        # Calculate loss (divide by accumulation steps)
        loss = F.nll_loss(y_pred, target) / accumulation_steps
        train_losses.append(loss * accumulation_steps)

        # Backpropagation
        loss.backward()

        # Update weights only after accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()*accumulation_steps} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

    return train_losses, train_acc


def test_loop(model, device, test_loader, test_losses, test_acc):
    """
    Test loop for one epoch
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))

    return test_losses, test_acc


