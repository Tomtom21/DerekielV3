from pathlib import Path
import sys
# Setting our working directory to the root of the project
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from training.lanenet.LaneDataset import LaneDataset
from training.lanenet.EarlyStop import EarlyStop
from training.lanenet.training_base import *
from models.lanenet.model import LaneNet
from models.lanenet import model

def main():
    args = arg_setup()
    args.dataset_dir = "data/processed_tusimple"  

    device = get_device()

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

    # Initiallizing EarlyStopping
    earlystop = EarlyStop(patience=args.patience, mode="loss")

    # Model, loss, optimizer
    model = LaneNet().to(device)
    if args.model_path is not None:
        print(f"Loading model weights from {args.model_path} to continue training...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    criterion_x = nn.MSELoss()
    criterion_vis = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Lists to store losses and accuracies for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    metric_gaps = []

    # Training loop
    epoch_iter = tqdm(range(args.epochs), desc="Epochs", unit="epoch")
    for epoch in epoch_iter:
        train_loss, train_acc = train_epoch(model, train_loader, criterion_x, criterion_vis, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion_x, criterion_vis, device)
        
        # Update tqdm bar with current metrics
        epoch_iter.set_postfix({
            "Train Loss": f"{train_loss:.4f}",
            "Train Acc": f"{train_acc:.4f}",
            "Val Loss": f"{val_loss:.4f}",
            "Val Acc": f"{val_acc:.4f}"
        })

        # Print metrics for each epoch
        print(f"Epoch [{epoch+1}/{args.epochs}], "
              f"Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Metric gap calculation
        metric_gaps.append(abs(train_acc - val_acc))

        # Early stopping check
        if earlystop(val_loss, model):
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
        elif earlystop.counter > 0:
            print(f"No improvement in validation loss for {earlystop.counter} epoch(s).")

        # Save the most recent model after each epoch
        torch.save(model.state_dict(), f"training/lanenet/model_files/lanenet_tusimple_last.pth")

    # Save the best model
    best_state = earlystop.get_best_model_state()
    if best_state is not None:
        torch.save(best_state, "training/lanenet/model_files/lanenet_tusimple_best.pth")

    # Plot ad save loss and accuracy graphs
    epochs = range(1, len(train_losses) + 1)
    generate_loss_acc_graph(epochs, train_losses, val_losses, train_accuracies, val_accuracies, metric_gaps)

if __name__ == "__main__":
    main()
