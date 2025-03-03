import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder

import torch
from sklearn.preprocessing import LabelEncoder



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# Load dataset
df = pd.read_csv(r'C:\Users\ikanh\Desktop\keggle\data\clean.csv')

# Feature Engineering
df['plate_letters'] = df['plate'].str.extract(r'([A-Z]+)')
df['plate_numbers'] = df['plate'].str.extract(r'(\d+)').astype(float)
df['plate_last_digit'] = df['plate'].str[-1]
df['plate_prefix'] = df['plate'].str[:1]
df['plate_region'] = df['plate'].str.extract(r'(\d{2,3})$').fillna(0).astype(int)

# Apply Target Encoding
target_encoder = TargetEncoder(cols=['plate_letters', 'plate_last_digit', 'plate_prefix', 'area'])
df[['plate_letters', 'plate_last_digit', 'plate_prefix', 'area']] = target_encoder.fit_transform(
    df[['plate_letters', 'plate_last_digit', 'plate_prefix', 'area']], df['price'])

# Log transform target variable
df['log_price'] = np.log1p(df['price'])

# Features & Target
X = df[['plate_letters', 'plate_numbers', 'plate_last_digit', 'plate_prefix', 'plate_region', 'area', 'year']].values
y = df['log_price'].values

# Standardize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMAPE function
def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

# Hyperparameter tuning with Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0)
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = np.expm1(model.predict(X_test))
    return smape(np.expm1(y_test), y_pred)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("Best Parameters:", best_params)

# Train XGBoost with optimized parameters
xgb_model = xgb.XGBRegressor(**best_params)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], )
y_pred_xgb = np.expm1(xgb_model.predict(X_test))
smape_xgb = smape(np.expm1(y_test), y_pred_xgb)
print(f"SMAPE (XGBoost): {smape_xgb:.2f}%")

# Train LightGBM
lgb_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=7)
lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],  )
y_pred_lgb = np.expm1(lgb_model.predict(X_test))
smape_lgb = smape(np.expm1(y_test), y_pred_lgb)
print(f"SMAPE (LightGBM): {smape_lgb:.2f}%")
