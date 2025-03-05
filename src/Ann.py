from DataLoader import train_loader, val_loader, test_loader, PositionalEncoding, vocab_size
import torch.nn as nn
import torch
from sub import SMAPELoss, mape_loss, to_device_helper
from tqdm import tqdm
import torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Ann(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()

        self.plate_embedding = nn.Embedding(vocab_size, d_model)
        self.advantage_on_road_embedding = nn.Embedding(2, d_model)
        self.significance_embedding = nn.Embedding(11, d_model)
        self.year_embedding = nn.Embedding(5, d_model)

        self.fc = nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 1)
)

        # Xavier Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, plates, advantages_on_road, significances, years):
        batch_size = plates.size(0)

        plates_emb = self.plate_embedding(plates)  # (batch_size, 9, d_model)
        advantages_on_road_emb = self.advantage_on_road_embedding(advantages_on_road).unsqueeze(1)  # (batch_size, 1, d_model)
        significances_emb = self.significance_embedding(significances).unsqueeze(1)
        years_emb = self.year_embedding(years).unsqueeze(1)

        # Concatenate embeddings (sequence length = 12)
        x = torch.cat([plates_emb, advantages_on_road_emb, significances_emb, years_emb], dim=1)  # (batch_size, 12, d_model)

        # Mean Pooling
        x = x.mean(dim=1)  # (batch_size, d_model)

        # Fully Connected Layer
        return self.fc(x)


def train_model(model, criterion, optimizer, train_loader, val_loader=None, epochs: int = 5, device: str = "cuda", scheduler=None):
    print(f"Training on {device}...")
    train_length = len(train_loader.dataset)
    val_length = len(val_loader.dataset) if val_loader is not None else 0

    for epoch in range(epochs):
        model.train()
        total_loss, train_mape_sum = 0.0, 0.0

        with tqdm(total=train_length, desc=f"Epoch [{epoch + 1}/{epochs}]", unit='batch') as pbar:
            for plates, advantages_on_road, significances, years, prices in train_loader:
                batch_size = plates.size(0)
                optimizer.zero_grad()

                plates, advantages_on_road, significances, years, prices = to_device_helper(
                    (plates, advantages_on_road, significances, years, prices), device
                )

                preds = model(plates, advantages_on_road, significances, years).squeeze()
                loss = criterion(preds, prices)
                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item() * batch_size
                train_mape_sum += mape_loss(preds, prices).item() * batch_size

                pbar.update(batch_size)

        train_avg_loss = total_loss / train_length
        train_avg_mape = train_mape_sum / train_length

        if val_loader is None:
            print(f"Train Loss: {train_avg_loss:.4f}, Train MAPE: {train_avg_mape:.2f}%")
            continue

        model.eval()
        val_loss, val_mape_sum = 0.0, 0.0

        with torch.no_grad():
            for plates, advantages_on_road, significances, years, prices in val_loader:
                batch_size = plates.size(0)
                plates, advantages_on_road, significances, years, prices = to_device_helper(
                    (plates, advantages_on_road, significances, years, prices), device
                )

                preds = model(plates, advantages_on_road, significances, years).squeeze()
                loss = criterion(preds, prices)

                val_loss += loss.item() * batch_size
                val_mape_sum += mape_loss(preds, prices).item() * batch_size

        val_avg_loss = val_loss / val_length
        val_avg_mape = val_mape_sum / val_length

        print(f"Epoch [{epoch + 1}/{epochs}] | "
              f"Train Loss: {train_avg_loss:.4f}, Train MAPE: {train_avg_mape:.2f}% | "
              f"Val Loss: {val_avg_loss:.4f}, Val MAPE: {val_avg_mape:.2f}%")

        if scheduler is not None:
            scheduler.step(val_avg_loss)  # Reduce LR on plateau


# Model Initialization
model = Ann(
    vocab_size=vocab_size,
    d_model=128,
    nhead=4,
    num_layers=2,
)
model.to(DEVICE)

criterion = SMAPELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

EPOCHS = 250
train_model(model, criterion, optimizer, train_loader, val_loader=val_loader, epochs=EPOCHS, device=DEVICE, scheduler=scheduler)

test_loss = 0.0
