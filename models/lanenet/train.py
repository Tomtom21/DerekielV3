import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms
from LaneDataset import LaneDataset
from model import LaneNet
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

def compute_accuracy(outputs, targets, threshold=0.1):
    """
    Computes accuracy as the percentage of keypoints (visibility==1) where predicted x is within threshold of target x.
    """
    # outputs, targets: [batch, 40]
    outputs = outputs.view(-1, 20, 2)
    targets = targets.view(-1, 20, 2)
    vis_mask = targets[:, :, 0] == 1  # Only consider visible keypoints
    if vis_mask.sum() == 0:
        return 0.0
    pred_x = outputs[:, :, 1]
    true_x = targets[:, :, 1]
    correct = ((torch.abs(pred_x - true_x) < threshold) & vis_mask).sum().item()
    total = vis_mask.sum().item()
    return correct / total

def moving_average(data, window_size=5):
    if len(data) < window_size:
        return data
    return [sum(data[max(0, i-window_size+1):i+1]) / (i - max(0, i-window_size+1) + 1) for i in range(len(data))]

def main():
    parser = argparse.ArgumentParser(description="Train LaneNet model")
    parser.add_argument('--dataset_dir', type=str, default='../../data/lanenet_dataset', help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of data to use for validation')
    args = parser.parse_args()

    # Device selection: MPS > CUDA > CPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Image transformations (add more as needed)
    transform = transforms.Compose([
        transforms.Resize((128, 227)),
        transforms.ToTensor(),
    ])

    # Dataset
    full_dataset = LaneDataset(args.dataset_dir, transform=transform)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model, loss, optimizer
    model = LaneNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Lists to store losses and accuracies for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    metric_gaps = []

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]", leave=False)
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            acc = compute_accuracy(outputs.detach().cpu(), targets.detach().cpu())
            running_acc += acc * images.size(0)
            progress_bar.set_postfix(loss=loss.item(), acc=acc)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_acc / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                acc = compute_accuracy(outputs.cpu(), targets.cpu())
                val_acc += acc * images.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{args.epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Metric gap (absolute difference in accuracy)
        metric_gaps.append(abs(epoch_acc - val_acc))

    # Save the model
    torch.save(model.state_dict(), "lanenet.pth")
    print("Training complete. Model saved as lanenet.pth.")

    # Plot and save loss and accuracy graphs
    epochs = range(1, args.epochs + 1)
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

if __name__ == "__main__":
    main()
