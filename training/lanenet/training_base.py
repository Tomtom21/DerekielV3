import torch
import argparse
from matplotlib import pyplot as plt


def arg_setup():
    parser = argparse.ArgumentParser(description="Train LaneNet model")
    parser.add_argument('--dataset_dir', type=str, default='../../data/lanenet_dataset', help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of data to use for validation')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (epochs)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model weights to continue training')
    args = parser.parse_args()
    return args

def get_device():
    # Device selection: MPS > CUDA > CPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def generate_loss_acc_graph(epochs, train_losses, val_losses, train_accuracies, val_accuracies, metric_gaps):
    """
    Generates and saves a graph of training/validation loss and accuracy over epochs, including metric gap.
    
    :param epochs: Range of epoch numbers
    :param train_losses: List of training loss values per epoch
    :param val_losses: List of validation loss values per epoch
    :param train_accuracies: List of training accuracy values per epoch
    :param val_accuracies: List of validation accuracy values per epoch
    :param metric_gaps: List of metric gap values (absolute difference between training and validation accuracy) per epoch
    """
    def moving_average(data, window_size=5):
        if len(data) < window_size:
            return data
        return [sum(data[max(0, i-window_size+1):i+1]) / (i - max(0, i-window_size+1) + 1) for i in range(len(data))]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Compute moving averages
    train_acc_ma = moving_average(train_accuracies)
    val_acc_ma = moving_average(val_accuracies)

    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.plot(epochs, metric_gaps, label='Metric Gap (|Train Acc - Val Acc|)')
    plt.plot(epochs, train_acc_ma, '--', label='Train Acc (Moving Avg)')
    plt.plot(epochs, val_acc_ma, '--', label='Val Acc (Moving Avg)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy / Gap')
    plt.title('Training vs. Validation Accuracy and Metric Gap')
    plt.legend()

    plt.tight_layout()
    plt.savefig("loss_accuracy_graph.png")
    plt.close()
    print("Loss and accuracy graph saved as loss_accuracy_graph.png.")

def train_epoch(model, train_loader, criterion_x, criterion_vis, optimizer, device):
    """
    Training the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    # Looping through the batches
    for images, x_targets, vis_targets in train_loader:
        # Moving the image/data to the device
        images = images.to(device)
        x_targets = x_targets.to(device)
        vis_targets = vis_targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        pred_x, pred_vis_logits = model(images)
        
        # Calculate loss/accuracy
        loss = calculate_loss(
            pred_x, pred_vis_logits, x_targets, vis_targets, criterion_x, criterion_vis
        )
        acc = calculate_accuracy(
            pred_x.detach().cpu(), x_targets.detach().cpu(), vis_targets.detach().cpu()
        )

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update running loss and accuracy
        running_loss += loss.item() * images.size(0)
        running_acc += acc * images.size(0)

    # Calculating epoch loss and accuracy
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_acc / len(train_loader.dataset)
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion_x, criterion_vis, device):
    """
    Validating the model for one epoch.
    """
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    
    # Looping through the validation batches
    with torch.no_grad():
        for images, x_targets, vis_targets in val_loader:
            # Moving the image/data to the device
            images = images.to(device)
            x_targets = x_targets.to(device)
            vis_targets = vis_targets.to(device)

            # Forward pass
            pred_x, pred_vis_logits = model(images)

            # Calculate loss/accuracy
            loss = calculate_loss(
                pred_x, pred_vis_logits, x_targets, vis_targets, criterion_x, criterion_vis
            )
            val_loss += loss.item() * images.size(0)
            acc = calculate_accuracy(
                pred_x.detach().cpu(), x_targets.detach().cpu(), vis_targets.detach().cpu()
            )

            # Update running accuracy
            val_acc += acc * images.size(0)

    # Calculating the validation loss and accuracy for the epoch
    val_loss /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)
    return val_loss, val_acc

def calculate_loss(
        pred_x, 
        pred_vis_logits, 
        x_targets, 
        vis_targets, 
        criterion_x, 
        criterion_vis,
        x_weight=15.0,
        vis_weight=1.0):
    """
    Calculates the combined loss for x position and visibility.
    """
    loss_vis = criterion_vis(pred_vis_logits, vis_targets)
    vis_mask = vis_targets == 1
    loss_x = criterion_x(pred_x[vis_mask], x_targets[vis_mask])
    # print(f"Loss_x: {loss_x.item():.4f}, Loss_vis: {loss_vis.item():.4f}")

    return (x_weight * loss_x) + (vis_weight * loss_vis)

def calculate_accuracy(pred_x, x_targets, vis_targets, threshold=0.1):
    """
    Computes accuracy as the percentage of keypoints (visibility==1) where predicted x is within threshold of target x.
    """

    # Take the compute_accuracy function and implement it in here
    vis_mask = vis_targets == 1  # Only consider visible keypoints
    if vis_mask.sum() == 0:
        return 0.0
    correct = ((torch.abs(pred_x - x_targets) < threshold) & vis_mask).sum().item()
    total = vis_mask.sum().item()
    return correct / total
