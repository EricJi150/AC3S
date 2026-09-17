import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ac3s.dataset import ModulatorDataset
from ac3s.model import Modulator


def evaluate(model, dataloader, device):
    model.eval()
    y_list, y_pred_list = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_list.append(y)
            y_pred_list.append(model(X))
    return torch.cat(y_list, dim=0), torch.cat(y_pred_list, dim=0)


def main(
    data_file,
    output_dir,
    output_name,
    num_epochs,
    batch_size,
    learning_rate,
    lr_step_size,
    lr_gamma,
    train_split_ratio,
    device,
):
    # Set up training and validation dataloaders
    dataset = ModulatorDataset(data_file)
    print(f"Loaded {len(dataset)} samples, input dim {dataset.magnitudes.shape[1]}")

    X_mean = dataset.magnitudes.mean(dim=0)
    X_std = dataset.magnitudes.std(dim=0)
    dataset.magnitudes = (dataset.magnitudes - X_mean) / X_std

    num_train = int(len(dataset) * train_split_ratio)
    train_dataset = Subset(dataset, range(num_train))
    val_dataset = Subset(dataset, range(num_train, len(dataset)))
    y_train = dataset.conditioning_scales[:num_train]
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Initiazlize model
    model = Modulator().to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name + ".pt")

    # Training loop
    min_loss = float("inf")
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            loss = loss_fn(y_pred, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        # Validation
        y_list, y_pred_list = evaluate(model, val_dataloader, device)
        val_loss = loss_fn(y_pred_list, y_list.float()).item()
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save({"model_state": model.state_dict(), "mean": X_mean, "std": X_std}, output_path)
            print(f"Lower val loss @ epoch {epoch}: {val_loss:.4f}")

    print("Training complete!")

    # Validation using best checkpoint
    model.load_state_dict(torch.load(output_path, weights_only=True)["model_state"])
    y_list, y_pred_list = evaluate(model, val_dataloader, device)

    mae = torch.abs(y_list - y_pred_list)
    print("Val MAE:", mae.mean().item())
    print("Baseline MAE:", torch.abs(y_list - y_train.mean()).mean().item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/modulator_data.pt", help="Path to the modulator data .pt file")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save the trained modulator checkpoint in")
    parser.add_argument("--output_name", type=str, default="modulator_ckpt", help="Filename (without extension) to save the checkpoint as")
    parser.add_argument("--num_epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training/testing batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Adam learning rate")
    parser.add_argument("--lr_step_size", type=int, default=20, help="StepLR step size")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="StepLR decay factor")
    parser.add_argument("--train_split_ratio", type=float, default=0.8, help="Ratio of data used for training")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    args = parser.parse_args()

    main(
        args.data_file,
        args.output_dir,
        args.output_name,
        args.num_epochs,
        args.batch_size,
        args.learning_rate,
        args.lr_step_size,
        args.lr_gamma,
        args.train_split_ratio,
        args.device,
    )
