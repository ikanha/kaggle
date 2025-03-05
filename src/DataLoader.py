import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
import re
import string
from tqdm import tqdm
from supplemental_english import GOVERNMENT_CODES
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 37
PLATE_POSSIBLE_LETTERS = "ABEKMHOPCTYX"  # 12 total
ALL_CHARS = PLATE_POSSIBLE_LETTERS + string.digits  # 12 + 10 = 22 total

char2idx = {c: i for i, c in enumerate(ALL_CHARS)}  # char to identifier map
vocab_size = len(char2idx)

data = pd.read_csv(
    r"C:\Users\ikanh\Desktop\keggle\data\train.csv",
    dtype={
        "id": int,
        "plate": str,
        "price": float,
    },
    parse_dates=["date"],
)


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



def data_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    # leaving only years (2021 -> 0, 2022 -> 1, etc.)
    data["year"] = data["date"].dt.year - 2021

    # adding features (advantage on road (bool), significance (int))
    data = add_advantage_on_road_and_significance(data)

    # adding 0 if a plate's region code is only 2 digits long
    data["plate"] = data["plate"].apply(lambda plate: plate if len(plate) == 9 else f"{plate[:6]}0{plate[6:]}")

    # removing unnecessary
    data = data.drop(columns=["id", "date"])
    return data

data = data.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
data = data_preprocessing(data)

class CarPlateDataset(Dataset):
    def __init__(self, data:pd.DataFrame, char2idx:dict, return_id: bool = False):
        self.data = data
        self.char2idx = char2idx
        self.return_id = return_id


    def __len__(self):
        return len(self.data)
    
    def encode_plate(self, plate: str) -> list[int]:
        encoded = []
        
        for char in plate:
            if char in self.char2idx:
                encoded.append(self.char2idx[char])
            else:
                encoded.append(0)
        return encoded

    def __getitem__(self, idx)->tuple:
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



