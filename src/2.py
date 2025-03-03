import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch

# Check for CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# Load dataset
df = pd.read_csv(r'C:\Users\ikanh\Desktop\keggle\data\clean.csv')

# Feature Engineering
df['plate_letters'] = df['plate'].str.extract(r'([A-Z]+)')  # Extract letters
df['plate_numbers'] = df['plate'].str.extract(r'(\d+)').astype(float)  # Extract numbers
df['plate_last_digit'] = df['plate'].str[-1].astype(str)  # Extract last character

# Encode categorical features
le_area = LabelEncoder()
df['area'] = le_area.fit_transform(df['area'])

le_plate_letters = LabelEncoder()
df['plate_letters'] = le_plate_letters.fit_transform(df['plate_letters'])

le_plate_last_digit = LabelEncoder()
df['plate_last_digit'] = le_plate_last_digit.fit_transform(df['plate_last_digit'])

# Apply log transformation to target variable (price)
df['log_price'] = np.log1p(df['price'])

# Features & target
X = df[['plate_letters', 'plate_numbers', 'plate_last_digit', 'area', 'year']].values
y = df['log_price'].values

# Standardize numeric features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Best Parameters from Optuna Optimization
best_params = {
    'n_estimators': 529,
    'max_depth': 10,
    'learning_rate': 0.03631779324526718,
    'subsample': 0.8912907721274288,
    'colsample_bytree': 0.9241638084736049,
    'reg_alpha': 0.21271622870093088,
    'reg_lambda': 0.9633405819984413,
    'random_state': 42
}

# XGBoost model with optimized parameters
xgb_model = xgb.XGBRegressor(**best_params)

# Train model
xgb_model.fit(X_train, y_train)

# Predictions (convert back from log scale)
y_pred_xgb = np.expm1(xgb_model.predict(X_test))

# Reverse log transformation on actual prices
y_test_actual = np.expm1(y_test)

# SMAPE function
def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

# Calculate SMAPE
smape_xgb = smape(y_test_actual, y_pred_xgb)
print(f"SMAPE (XGBoost): {smape_xgb:.2f}%")
