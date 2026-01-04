"""
Utilities for saving and loading models.
"""
import torch
import os

def save_checkpoint(model, epoch, output_dir, filename=None):
    """
    Saves the model weights to the specified directory.
    
    Args:
        model (nn.Module): The PyTorch model to save.
        epoch (int): The current epoch number.
        output_dir (str): Directory where the model will be saved.
        filename (str, optional): Custom filename. Defaults to resnet50_epoch_X.pth.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Folder created : {output_dir}")

    if filename is None:
        filename = f"resnet50_reid_epoch_{epoch}.pth"
    
    save_path = os.path.join(output_dir, filename)

    torch.save(model.state_dict(), save_path)
    print(f"Model Saved : {save_path}")


def load_checkpoint(model, checkpoint_path, device='cpu'):
    """
    Loads weights from a .pth file into the model.
    
    Args:
        model (nn.Module): The empty model architecture.
        checkpoint_path (str): Path to the .pth file.
        device (str): Device to load the model on ('cpu' or 'cuda').
        
    Returns:
        nn.Module: The model with loaded weights, set to eval mode.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No such checkpoint found at: {checkpoint_path}")

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=torch.device(device))
    
    # Inject weigths
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()

    print(f"Modèle succesfully loaded from : {checkpoint_path}")
    return model