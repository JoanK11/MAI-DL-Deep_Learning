import torch, os
from tqdm import tqdm

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path, map_location=None):
    state = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state)
    return model

def epoch_loop(model, loader, criterion, optimizer=None, device='cuda'):
    running_loss, correct, total = 0.0, 0, 0
    model.train(optimizer is not None)
    for X, y in tqdm(loader, leave=False):
        X, y = X.to(device), y.to(device)
        if optimizer:
            optimizer.zero_grad()
        out = model(X)
        loss = criterion(out if not isinstance(out, tuple) else out[0], y)
        if optimizer:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * X.size(0)
        if not isinstance(out, tuple):
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    avg_loss = running_loss / len(loader.dataset)
    acc = correct / total if total else None
    return avg_loss, acc

def epoch_loop_autoencoder(model, loader, criterion, optimizer=None, device='cuda'):
    running_loss = 0.0
    model.train(optimizer is not None)
    for X, _ in tqdm(loader, leave=False):
        X = X.to(device)
        target = X
        if optimizer:
            optimizer.zero_grad()
        out = model(X)
        loss = criterion(out if not isinstance(out, tuple) else out[0], target)
        if optimizer:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * X.size(0)
    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, None
