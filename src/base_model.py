import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
# Load dataset
df = pd.read_csv(r'C:\Users\ikanh\Desktop\keggle\data\clean.csv')

# Feature Engineering: Extract info from plate numbers
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
y = df['log_price'].values  # Using log-transformed price

# Standardize numeric features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model with optimized parameters
xgb_model = xgb.XGBRegressor(
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=5, 
    subsample=0.9,  
    colsample_bytree=0.9,  
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=0.5,  # L2 regularization
    random_state=42
)

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