from DataLoader import train_loader, val_loader, test_loader, PositionalEncoding, vocab_size
import torch.nn as nn
import torch
from sub import mape_loss, to_device_helper
from tqdm import tqdm
import torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Ann(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 4):
        super().__init__()

        self.plate_embedding = nn.Embedding(vocab_size, d_model)
        self.advantage_on_road_embedding = nn.Embedding(2, d_model)
        self.significance_embedding = nn.Embedding(11, d_model)
        self.year_embedding = nn.Embedding(5, d_model)

        # Transformer Encoder for better feature extraction
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=512, dropout=0.3, 
                                                   activation="gelu", batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

        # Xavier Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, plates, advantages_on_road, significances, years):
        plates_emb = self.plate_embedding(plates)  
        advantages_on_road_emb = self.advantage_on_road_embedding(advantages_on_road).unsqueeze(1)
        significances_emb = self.significance_embedding(significances).unsqueeze(1)
        years_emb = self.year_embedding(years).unsqueeze(1)

        x = torch.cat([plates_emb, advantages_on_road_emb, significances_emb, years_emb], dim=1)

        # Apply Transformer Encoder
        x = self.transformer_encoder(x)

        # Mean Pooling
        x = x.mean(dim=1)

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

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)  # Gradient Clipping

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
            scheduler.step(val_avg_loss)


# Model Initialization
model = Ann(
    vocab_size=vocab_size,
    d_model=256,  # Increased embedding size
    nhead=4,
    num_layers=4  # More transformer layers for deeper feature learning
)
model.to(DEVICE)

criterion = mape_loss  # Keeping the original MAPE loss function
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)  # Lower LR for better convergence
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

EPOCHS = 250
train_model(model, criterion, optimizer, train_loader, val_loader=val_loader, epochs=EPOCHS, device=DEVICE, scheduler=scheduler)
 
 
 # save the model
torch.save(model.state_dict(), '01model.pth')