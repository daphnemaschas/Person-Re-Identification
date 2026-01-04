"""
Training logic for Person Re-Identification.
"""
import os
import torch
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion_xent, criterion_triplet, device, config):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: get both classification scores and raw features
        logits, features = model(images)
        
        # Compute losses based on your config weights
        loss_xent = criterion_xent(logits, labels)
        loss_triplet = criterion_triplet(features, labels)
        
        total_loss = (config['weight_xent'] * loss_xent + 
                      config['weight_triplet'] * loss_triplet)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
        
    return running_loss / len(loader)


def train_model(model, train_loader, optimizer, scheduler, criterion_xent, criterion_triplet, device, config, num_epochs, output_dir):
    """
    Executes the full training loop.
    
    Args:
        model: The ResNet50ReID model.
        train_loader: DataLoader for the training set.
        optimizer: Optimization algorithm (Adam).
        scheduler: Learning rate scheduler.
        criterion_xent: CrossEntropy loss function.
        criterion_triplet: Triplet loss function.
        device: 'cpu' or 'cuda'.
        config: Loss weights from YAML.
        num_epochs: Number of epochs to run.
        output_dir: Path to save the model weights.
    """
    # Create directory for saving models
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # tqdm bar for the current epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: get logits (for Xent) and features (for Triplet)
            logits, features = model(images)
            
            # Compute losses
            loss_xent = criterion_xent(logits, labels)
            loss_triplet = criterion_triplet(features, labels)
            
            # Combine losses
            total_loss = (config['weight_xent'] * loss_xent + 
                          config['weight_triplet'] * loss_triplet)
            
            # Optimization
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            pbar.set_postfix({
                'loss': f"{running_loss / (pbar.n + 1):.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
            
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint after each epoch
        save_path = os.path.join(output_dir, f"resnet50_latest.pth")
        torch.save(model.state_dict(), save_path)
        
    print(f"\nTraining finished! Model saved in {output_dir}")