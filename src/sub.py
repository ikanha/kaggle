import os
import sys
import string
from tqdm import tqdm

import pandas as pd
from supplemental_english import *  # REGION_CODES, GOVERNMENT_CODES

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch

import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


def to_device_helper(objects: tuple, device: str) -> tuple:
    return tuple([obj.to(device) for obj in objects])


class CarPlateDataset(Dataset):
    def __init__(self, data, char2idx: dict, return_id: bool = False):
        self.data = data
        self.char2idx = char2idx
        self.return_id = return_id

    def __len__(self) -> int:
        return len(self.data)

    def encode_plate(self, plate: str) -> list[int]:
        encoded = []
        
        for char in plate:
            if char in self.char2idx:
                encoded.append(self.char2idx[char])
            else:
                encoded.append(0)
        return encoded

    def __getitem__(self, idx) -> tuple:
        plate_string, advantage_on_road, significance, year, price = self.data.loc[idx, [
            "plate", "advantage_on_road", "significance", "year", "price"
        ]].to_list()
        result = (self.encode_plate(plate_string), advantage_on_road, significance, year, price)

        if self.return_id:
            return (self.data["id"][idx], *result)
        else:
            return result


def collate_fn(batch):
    plates, advantages_on_road, significances, years, prices = [], [], [], [], []

    for (plate, advantage_on_road, significance, year, price) in batch:
        plates.append(plate)
        advantages_on_road.append(advantage_on_road)
        significances.append(significance)
        years.append(year)
        prices.append(price)

    plates              = torch.tensor(plates,              dtype=torch.long)
    advantages_on_road  = torch.tensor(advantages_on_road,  dtype=torch.long)
    significances       = torch.tensor(significances,       dtype=torch.long)
    years               = torch.tensor(years,               dtype=torch.long)
    prices              = torch.tensor(prices,              dtype=torch.float)

    return plates, advantages_on_road, significances, years, prices


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class SMAPELoss(nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        numerator = 2.0 * (pred - target).abs()
        denominator = pred.abs() + target.abs() + self.epsilon
        return 100.0 * torch.mean(numerator / denominator)


def mape_loss(y_pred, y_true, epsilon: float = 1e-8):
    numerator = (y_true - y_pred).abs()
    denominator = (y_true.abs() + epsilon)
    return 100.0 * torch.mean(numerator / denominator)


# class SimpleTransformer(nn.Module):
#     def __init__(self, vocab_size: int, d_model: int = 48, nhead: int = 4, num_layers: int = 2):
#         super().__init__()

#         # embeddings
#         self.plate_embedding              = nn.Embedding(vocab_size,  d_model)
#         self.advantage_on_road_embedding  = nn.Embedding(2,           d_model)  # 0 or 1
#         self.significance_embedding       = nn.Embedding(11,          d_model)  # from 0 to 10
#         self.year_embedding               = nn.Embedding(5,           d_model)  # from 2021 to 2025 (0 to 4 in the dataset)

#         # Modify positional embedding to account for additional factors
#         self.pos_embedding                = nn.Embedding(16,          d_model)  # (9 chars + 1 bool + 2 int + 4 additional factors)

#         # encoder
#         self.transformer_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
#             num_layers=num_layers
#         )

#         # full-connected, output
#         self.fc = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, 128),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )

#     def forward(self, plates, advantages_on_road, significances, years):
#         batch_size = plates.size(0)

#         # embedding each input
#         plates_emb                = self.plate_embedding(plates)                                        # (batch_size, 9, d_model)
#         advantages_on_road_emb    = self.advantage_on_road_embedding(advantages_on_road).unsqueeze(1)   # (batch_size, 1, d_model)
#         significances_emb         = self.significance_embedding(significances).unsqueeze(1)             # (batch_size, 1, d_model)
#         years_emb                 = self.year_embedding(years).unsqueeze(1)                             # (batch_size, 1, d_model)

#         # concatenating along the sequence dimension -> total length = 9 + 1 + 1 + 1 + 4 = 16
#         x = torch.cat([
#             plates_emb,
#             advantages_on_road_emb,
#             significances_emb,
#             years_emb,
#             # Add embeddings for the additional factors here
#         ], dim=1)  # (batch_size, 16, d_model)

#         # adding positional embeddings
#         positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand(batch_size, -1)  # (batch_size, 16)
#         x = self.pos_embedding(positions)  # (batch_size, 16, d_model)

#         # transformer operation itself
#         x = self.transformer_encoder(x)  # (batch_size, 16, d_model)

#         # mean pooling
#         x = x.mean(dim=1)  # (batch_size, d_model)

#         # applying the final full-connected layer
#         return self.fc(x)

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 48, nhead: int = 4, num_layers: int = 2):
        super().__init__()

        # Embeddings
        self.plate_embedding              = nn.Embedding(vocab_size,  d_model)
        self.advantage_on_road_embedding  = nn.Embedding(2,           d_model)  # 0 or 1
        self.significance_embedding       = nn.Embedding(11,          d_model)  # 0 to 10
        self.year_embedding               = nn.Embedding(5,           d_model)  # 2021 to 2025 (mapped to 0-4)

        # Positional Encoding (Using the Modified Version)
        self.positional_encoding          = PositionalEncoding(d_model, max_len=16)

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, plates, advantages_on_road, significances, years):
        batch_size = plates.size(0)

        # Embedding Lookup
        plates_emb                = self.plate_embedding(plates)                                        # (batch_size, 9, d_model)
        advantages_on_road_emb    = self.advantage_on_road_embedding(advantages_on_road).unsqueeze(1)   # (batch_size, 1, d_model)
        significances_emb         = self.significance_embedding(significances).unsqueeze(1)             # (batch_size, 1, d_model)
        years_emb                 = self.year_embedding(years).unsqueeze(1)                             # (batch_size, 1, d_model)

        # Concatenating Along Sequence Dimension → Total Sequence Length = 9 + 1 + 1 + 1 = 12
        x = torch.cat([plates_emb, advantages_on_road_emb, significances_emb, years_emb], dim=1)  # (batch_size, 12, d_model)

        # Applying Positional Encoding
        x = self.positional_encoding(x)

        # Transformer Encoding
        x = self.transformer_encoder(x)  # (batch_size, 12, d_model)

        # Mean Pooling
        x = x.mean(dim=1)  # (batch_size, d_model)

        # Fully Connected Layer
        return self.fc(x)

def train_model(model, criterion, optimizer, train_loader, val_loader=None, epochs: int = 5, device: str = "cuda", scheduler=None):
    print(f"Training on {device}...")
    train_length = len(train_loader.dataset)
    val_length = len(val_loader.dataset) if val_loader is not None else 0

    for epoch in range(epochs):
        # training
        model.train()

        total_loss = 0.0
        train_mape_sum = 0.0

        # Wrap the training loop with tqdm
        with tqdm(total=train_length, desc=f"Epoch [{epoch + 1}/{epochs}]", unit='batch') as pbar:
            for plates, advantages_on_road, significances, years, prices in train_loader:
                batch_size = plates.size(0)

                optimizer.zero_grad()

                plates, advantages_on_road, significances, years, prices = to_device_helper(
                    (plates, advantages_on_road, significances, years, prices),
                    device
                )

                preds = model(plates, advantages_on_road, significances, years).squeeze()  # (batch_size,)
                loss = criterion(preds, prices)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_size
                train_mape_sum += mape_loss(preds, prices).item() * batch_size

                # Update the progress bar
                pbar.update(batch_size)

        train_avg_loss = total_loss / train_length
        train_avg_mape = train_mape_sum / train_length

        if val_loader is None:
            print(f"Train Loss: {train_avg_loss:.4f}, Train MAPE: {train_avg_mape:.2f}%")
            continue

        # validation
        model.eval()

        val_loss = 0.0
        val_mape_sum = 0.0

        with torch.no_grad():
            for plates, advantages_on_road, significances, years, prices in val_loader:
                batch_size = plates.size(0)

                plates, advantages_on_road, significances, years, prices = to_device_helper(
                    (plates, advantages_on_road, significances, years, prices),
                    device
                )

                preds = model(plates, advantages_on_road, significances, years).squeeze()  # (batch_size,)
                loss = criterion(preds, prices)

                val_loss += loss.item() * batch_size
                val_mape_sum += mape_loss(preds, prices).item() * batch_size

        val_avg_loss = val_loss / val_length
        val_avg_mape = val_mape_sum / val_length

        print(f"Epoch [{epoch + 1}/{epochs}] | "
              f"Train Loss: {train_avg_loss:.4f}, Train MAPE: {train_avg_mape:.2f}% | "
              f"Val Loss: {val_avg_loss:.4f}, Val MAPE: {val_avg_mape:.2f}%")

        if scheduler is not None:
            scheduler.step()

RANDOM_STATE = 37
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


PLATE_POSSIBLE_LETTERS = "ABEKMHOPCTYX"  # 12 total
ALL_CHARS = PLATE_POSSIBLE_LETTERS + string.digits  # 12 + 10 = 22 total

char2idx = {c: i for i, c in enumerate(ALL_CHARS)}  # char to identifier map
vocab_size = len(char2idx)





def covert_government_codes(gov_codes: dict) -> dict:
    result_gov_codes = {}

    for plate_range, importance_values in gov_codes.items():
        key = (plate_range[0], plate_range[2])
        value = {
            "numbers_range": plate_range[1],
            "importance_values": importance_values,
        }

        if key in result_gov_codes:
            result_gov_codes[key].append(value)
        else:
            result_gov_codes[key] = [value]

    return result_gov_codes


GOVERNMENT_CODES = covert_government_codes(GOVERNMENT_CODES)


def find_importance_values_for_plate(plate: str, gov_codes: dict) -> tuple:
    letters = plate[0] + plate[4:6]
    numbers = int(plate[1:4])
    region_code = plate[6:]

    key = (letters, region_code)
    if key in gov_codes:
        for plates_range in gov_codes[key]:  # iterating over ranges of numbers
            if plates_range["numbers_range"][0] <= numbers <= plates_range["numbers_range"][1]:  # found
                return (plates_range["importance_values"][2], plates_range["importance_values"][3])

    return (0, 0)  # ordinary plate, without any government affiliation


def add_advantage_on_road_and_significance(data: pd.DataFrame) -> pd.DataFrame:
    def apply_helper(row):
        advantage_on_road, significance = find_importance_values_for_plate(row["plate"], GOVERNMENT_CODES)
        return pd.Series({
            "advantage_on_road": advantage_on_road,
            "significance": significance,
        })

    data[["advantage_on_road", "significance"]] = data.apply(apply_helper, axis=1)
    return data


data = pd.read_csv(
    r"C:\Users\ikanh\Desktop\keggle\data\train.csv",
    dtype={
        "id": int,
        "plate": str,
        "price": float,
    },
    parse_dates=["date"],
)

# leaving only years (2021 -> 0, 2022 -> 1, etc.)
data["year"] = data["date"].dt.year - 2021

# adding features (advantage on road (bool), significance (int))
data = add_advantage_on_road_and_significance(data)

# adding 0 if a plate's region code is only 2 digits long
data["plate"] = data["plate"].apply(lambda plate: plate if len(plate) == 9 else f"{plate[:6]}0{plate[6:]}")

# removing unnecessary
data = data.drop(columns=["id", "date"])


data = data.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
dataset = CarPlateDataset(data, char2idx)

total_dataset_length = len(dataset)

train_size = int(0.8 * total_dataset_length)
val_size   = int(0.1 * total_dataset_length)
test_size  = total_dataset_length - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
)





train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, collate_fn=collate_fn)



# model = SimpleTransformer(
#     vocab_size=vocab_size,
#     d_model=128,
#     nhead=4,
#     num_layers=8,
# )
# model.to(DEVICE)

model = SimpleTransformer(
    vocab_size=vocab_size,
    d_model=48,
    nhead=4,
    num_layers=2,
)
model.to(DEVICE)

# lr=1e-3
criterion = SMAPELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)



EPOCHS = 250
train_model(model, criterion, optimizer, train_loader, val_loader=val_loader, epochs=EPOCHS, device=DEVICE, scheduler=scheduler)


model.eval()
criterion = SMAPELoss()

test_loss = 0.0

with torch.no_grad():
    for plates, advantages_on_road, significances, years, prices in test_loader:
        batch_size = plates.size(0)

        plates, advantages_on_road, significances, years, prices = to_device_helper(
            (plates, advantages_on_road, significances, years, prices),
            DEVICE
        )

        preds = model(plates, advantages_on_road, significances, years).squeeze()
        loss = criterion(preds, prices)
        test_loss += loss.item() * batch_size

test_avg_loss = test_loss / len(test_loader.dataset)
print(f"\nFinal test SMAPE: {test_avg_loss:.4f}")



#Predicting test dataset:

test_data = pd.read_csv(
    r"C:\Users\ikanh\Desktop\keggle\data\test.csv",
    dtype={
        "id": int,
        "plate": str,
    },
    parse_dates=["date"],
)

# leaving only years (2021 -> 0, 2022 -> 1, etc.)
test_data["year"] = test_data["date"].dt.year - 2021

# adding features (advantage on road (bool), significance (int))
test_data = add_advantage_on_road_and_significance(test_data)

# adding 0 if a plate's region code is only 2 digits long
test_data["plate"] = test_data["plate"].apply(lambda plate: plate if len(plate) == 9 else f"{plate[:6]}0{plate[6:]}")

# removing unnecessary
test_data = test_data.drop(columns=["date"])


inference_model = model


def encode_plate(plate: str) -> list[int]:
    encoded = []
    for char in plate:
        if char in char2idx:
            encoded.append(char2idx[char])
        else:
            encoded.append(0)
    return encoded


def predict(inference_model, plate_string: str, year: int) -> float:
    advantage_on_road, significance = find_importance_values_for_plate(plate_string, GOVERNMENT_CODES)

    plates              = torch.tensor([encode_plate(plate_string)], dtype=torch.long)
    advantages_on_road  = torch.tensor([advantage_on_road], dtype=torch.long)
    significances       = torch.tensor([significance], dtype=torch.long)
    years               = torch.tensor([year], dtype=torch.long)

    plates, advantages_on_road, significances, years = to_device_helper(
        (plates, advantages_on_road, significances, years),
        DEVICE
    )

    inference_model.eval()

    with torch.no_grad():
        preds = inference_model(plates, advantages_on_road, significances, years)
        predicted_price = preds.item()

    return predicted_price





results = {}

for row in tqdm(test_data.itertuples(index=False), total=len(test_data)):
    id, plate_string, year = row.id, row.plate, row.year
    predicted_price = predict(inference_model, plate_string, year)

    results[id] = predicted_price



def generate_submission(predicted_prices: dict, destination: str) -> None:
    header = "id,price\n"
    result_lines = [f"{id},{price}" for id, price in predicted_prices.items()]

    with open(destination, 'w') as file:
      file.write(header + '\n'.join(result_lines))


generate_submission(results, "submission.csv")