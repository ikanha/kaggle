from for_optuna import *
import optuna
from tqdm import tqdm
import csv

EPOCHS = 1

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    d_model = trial.suggest_int('d_model', 128, 512)
    nhead = trial.suggest_int('nhead', 2, 32)
    num_layers = trial.suggest_int('num_layers', 6, 1024)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    # Ensure d_model is divisible by nhead
    if d_model % nhead != 0:
        raise optuna.TrialPruned()

    # Create the model with suggested hyperparameters
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    )
    model.to(DEVICE)

    criterion = SMAPELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)

    # Train the model
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for plates, advantages_on_road, significances, years, prices in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}", leave=False):
            plates, advantages_on_road, significances, years, prices = to_device_helper(
                (plates, advantages_on_road, significances, years, prices),
                DEVICE
            )
            optimizer.zero_grad()
            preds = model(plates, advantages_on_road, significances, years).squeeze()
            loss = criterion(preds, prices)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * plates.size(0)
        scheduler.step()

    # Evaluate the model on the validation set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for plates, advantages_on_road, significances, years, prices in tqdm(val_loader, desc="Validation", leave=False):
            plates, advantages_on_road, significances, years, prices = to_device_helper(
                (plates, advantages_on_road, significances, years, prices),
                DEVICE
            )
            preds = model(plates, advantages_on_road, significances, years).squeeze()
            loss = criterion(preds, prices)
            val_loss += loss.item() * plates.size(0)

    val_avg_loss = val_loss / len(val_loader.dataset)
    return val_avg_loss

def main():
    # Run Optuna optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    # Get the best hyperparameters
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")

    # Save the best hyperparameters to a CSV file
    with open('best_hyperparameters.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Parameter', 'Value'])
        for key, value in best_params.items():
            writer.writerow([key, value])

    # Train the final model with the best hyperparameters
    best_model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=best_params['d_model'],
        nhead=best_params['nhead'],
        num_layers=best_params['num_layers'],
    )
    best_model.to(DEVICE)

    criterion = SMAPELoss()
    optimizer = optim.Adam(best_model.parameters(), lr=best_params['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)

    for epoch in range(EPOCHS):
        best_model.train()
        train_loss = 0.0
        for plates, advantages_on_road, significances, years, prices in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}", leave=False):
            plates, advantages_on_road, significances, years, prices = to_device_helper(
                (plates, advantages_on_road, significances, years, prices),
                DEVICE
            )
            optimizer.zero_grad()
            preds = best_model(plates, advantages_on_road, significances, years).squeeze()
            loss = criterion(preds, prices)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * plates.size(0)
        scheduler.step()

    # Predicting test dataset with the best model
    inference_model = best_model

    results = {}
    for row in tqdm(test_data.itertuples(index=False), total=len(test_data), desc="Prediction"):
        id, plate_string, year = row.id, row.plate, row.year
        predicted_price = predict(inference_model, plate_string, year)
        results[id] = predicted_price

    generate_submission(results, "submission.csv")

if __name__ == "__main__":
    main()