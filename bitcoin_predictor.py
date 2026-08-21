# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:13:41 2020

@author: Nathan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------------------
# 1. Load & basic cleaning
# -------------------------------------------------
df = pd.read_csv(r'file_path_here')          # replace with real path
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Keep only what we need
df = df[['Date', 'Close']].copy()

# Simple, non-leaking imputation for any missing closes
df['Close'] = df['Close'].ffill().bfill()

# -------------------------------------------------
# 2. Feature engineering (lags + target)
# -------------------------------------------------
prediction_days = 30

# Target: price 30 days ahead
df['Target'] = df['Close'].shift(-prediction_days)

# Simple lag features (much stronger than raw Close alone)
for lag in [1, 2, 3, 5, 10]:
    df[f'Close_lag_{lag}'] = df['Close'].shift(lag)

# Drop rows that contain NaNs created by shifting
df = df.dropna().reset_index(drop=True)

feature_cols = [c for c in df.columns if c.startswith('Close_lag_')]
X = df[feature_cols].values
y = df['Target'].values
dates = df['Date'].values

# -------------------------------------------------
# 3. Chronological split (NO shuffling)
# -------------------------------------------------
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_test = dates[split_idx:]

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# -------------------------------------------------
# 4. Scale features (critical for SVR-RBF)
# -------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -------------------------------------------------
# 5. Train SVR
# -------------------------------------------------
# Reasonable starting point after scaling; still worth tuning later
svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
svr.fit(X_train_scaled, y_train)

# -------------------------------------------------
# 6. Evaluate on the held-out future period
# -------------------------------------------------
y_pred = svr.predict(X_test_scaled)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\nTest MAE : {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R²  : {r2:.4f}")

# -------------------------------------------------
# 7. Plot actual vs predicted on the test set
# -------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(dates_test, y_test, label='Actual', linewidth=1.5)
plt.plot(dates_test, y_pred, label='Predicted', linewidth=1.5, alpha=0.8)
plt.title(f'SVR Forecast (horizon = {prediction_days} days)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 8. Forecast the next `prediction_days` prices
#    (using the most recent available lag features)
# -------------------------------------------------
last_features = df[feature_cols].iloc[-1:].values          # shape (1, n_features)
last_features_scaled = scaler.transform(last_features)

# Because the model predicts t+30 from the current feature vector,
# a single forward prediction gives the price 30 days from now.
# For a full path of the next 30 days one would need a recursive
# multi-step scheme or a different model; here we keep the original
