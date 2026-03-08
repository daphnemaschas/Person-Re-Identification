"""
Training logic for Person Re-Identification.
"""
import os
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion_xent, criterion_triplet, device, config):
    """
    Trains the model for one epoch.

    Args:
        model: The ResNet50ReID model.
        loader: DataLoader for the training set.
        optimizer: Optimization algorithm (Adam).
        criterion_xent: CrossEntropy loss function.
        criterion_triplet: Triplet loss function.
        device: 'cpu' or 'cuda'.
        config: Loss weights from YAML.
    """
    model.train()
    running_loss = 0.0
    scaler = GradScaler(enabled=device.type == 'cuda')
    
    pbar = tqdm(loader, desc="Training")
    for images, labels, _ in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=device.type == 'cuda'):
            logits, features = model(images)
            loss_xent = criterion_xent(logits, labels)
            loss_triplet = criterion_triplet(features, labels)
            total_loss = (config['weight_xent'] * loss_xent +
                          config['weight_triplet'] * loss_triplet)
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += total_loss.item()
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
        
    return running_loss / len(loader)


def train_model(model, train_loader, optimizer, scheduler, criterion_xent, criterion_triplet, device, config, num_epochs, output_dir, model_name='resnet50'):
    """
    Executes the full training loop.
    
    Args:
        model: The Re-ID model (ResNet50 or ViTReID).
        train_loader: DataLoader for the training set.
        optimizer: Optimization algorithm (Adam).
        scheduler: Learning rate scheduler.
        criterion_xent: CrossEntropy loss function.
        criterion_triplet: Triplet loss function.
        device: 'cpu' or 'cuda'.
        config: Loss weights from YAML.
        num_epochs: Number of epochs to run.
        output_dir: Path to save the model weights.
        model_name: Base name for checkpoint files (default: 'resnet50').
    """
    os.makedirs(output_dir, exist_ok=True)
    history = {'epoch': [], 'loss': [], 'lr': []}
    scaler = GradScaler(enabled=device.type == 'cuda')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=device.type == 'cuda'):
                logits, features = model(images)
                loss_xent = criterion_xent(logits, labels)
                loss_triplet = criterion_triplet(features, labels)
                total_loss = (config['weight_xent'] * loss_xent +
                              config['weight_triplet'] * loss_triplet)
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += total_loss.item()
            pbar.set_postfix({
                'loss': f"{running_loss / (pbar.n + 1):.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        
        epoch_loss = running_loss / len(train_loader)
        history['epoch'].append(epoch + 1)
        history['loss'].append(epoch_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        scheduler.step()
        
        save_path = os.path.join(output_dir, f"{model_name}_latest.pth")
        torch.save(model.state_dict(), save_path)
        
    print(f"\nTraining finished! Model saved in {output_dir}")
    return history