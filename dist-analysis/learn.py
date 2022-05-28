from typing import Set, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataloader import T, DataLoader

class AptDataset(Dataset):
    inputs: torch.Tensor
    labels: torch.Tensor

    def __init__(self, combined: np.ndarray, price_index: int) -> None:
        self.labels = torch.tensor(combined[:, price_index], dtype=torch.float32)
        self.inputs = torch.tensor(np.delete(combined, price_index, 1), dtype=torch.float32)
    
    def __len__(self) -> int:
        return self.inputs.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float]:
        return (self.inputs[index], self.labels[index])


@dataclass
class DataCollection:
    train: AptDataset
    validation: AptDataset
    input_dim: int
    min: torch.Tensor
    max: torch.Tensor
    train_names: Set[str]


def prepare_dataset(path: str) -> DataCollection:
    TRAIN_RATE = 0.75

    combined = pd.read_csv(path).sample(frac=1).reset_index()
    row_count = len(combined.index)
    train_names = set()

    for row in combined[combined.index < int(row_count*TRAIN_RATE)].itertuples():
        train_names.add(row.name)
    combined = combined.drop(["index", "name"], axis=1)
    input_dim = len(combined.columns) - 1 # exclude price
    print(f"input_dim: {input_dim}")
    price_index = combined.columns.get_loc("price")
    
    # Normalize by column
    combined = combined.to_numpy()
    min = combined.min(axis=0, keepdims=1)
    max = combined.max(axis=0, keepdims=1)
    combined = (combined - min) / (max - min)

    return DataCollection(
        AptDataset(combined[:int(row_count*TRAIN_RATE)], price_index), # Train
        AptDataset(combined[int(row_count*TRAIN_RATE):], price_index), # Validation
        input_dim,
        min,
        max,
        train_names
    )


class AptPredictor(nn.Module):
    layers: nn.Sequential

    def __init__(self, input_dim: int) -> None:
        super(AptPredictor, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim//2),
            nn.LeakyReLU(),

            nn.BatchNorm1d(input_dim//2),
            nn.Linear(input_dim//2, 1),
            nn.LeakyReLU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


if __name__ == "__main__":
    N_EPOCHS = 100

    datacoll = prepare_dataset("./combined_df.csv")
    train_loader = DataLoader(datacoll.train, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(datacoll.validation, batch_size=64, shuffle=False, num_workers=4)

    model = AptPredictor(datacoll.input_dim)
    optimizer = optim.AdamW(model.parameters())
    criterion = nn.MSELoss()

    train_loss_hist = []
    val_loss_hist = []

    for epoch in range(1, N_EPOCHS+1):
        model.train()
        train_loss = 0.0
        val_loss = 0.0

        for i, train_pair in enumerate(train_loader, 0):
            x, y = train_pair
            y = torch.unsqueeze(y, 1)
            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            for i, val_pair in enumerate(val_loader, 0):
                x, y = val_pair
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()

        print(f"{epoch}:\tTraining Loss: {train_loss/len(train_loader):.5E}, Validation Loss: {val_loss/len(val_loader):.5E}")
        train_loss_hist.append(train_loss/len(train_loader))
        val_loss_hist.append(val_loss/len(val_loader))

    with torch.no_grad():
        model.eval()
        print("Saving predicted points...")
        combined = pd.read_csv("./combined_df.csv")

        # Import data for model input/output
        model_data = combined.drop(["name"], axis=1)
        input_dim = len(model_data.columns) - 1 # exclude price
        price_index = model_data.columns.get_loc("price")

        # Normalize by column
        model_data = model_data.to_numpy()
        min = model_data.min(axis=0, keepdims=1)
        max = model_data.max(axis=0, keepdims=1)
        model_data = (model_data - min) / (max - min)

        price_min = min[0, price_index]
        price_max = max[0, price_index]

        model_data = torch.tensor(model_data)
        model_dataset = AptDataset(model_data, price_index)

        # Import coordinates
        x = combined["x"].values.tolist()
        y = combined["y"].values.tolist()
        name = combined["name"].values.tolist()
        price = []
        is_train = []
        
        for (i, (input, label)) in enumerate(model_dataset):
            input = torch.unsqueeze(input, 0)
            output = model(input)
            price.append((output * (price_max - price_min) + price_min)[0, 0].item())
            is_train.append(name[i] in datacoll.train_names)
        
        result = pd.DataFrame(data={"name": name, "x": x, "y": y, "price": price, "is_train": is_train})
        result.to_csv("predicted_df.csv")

    plt.plot(range(0, len(train_loss_hist)), train_loss_hist, "r")
    plt.plot(range(0, len(val_loss_hist)), val_loss_hist, "b")
    plt.show()