"""main.py – XGBoost classification of ±0.2 % next‑day moves.
Usage:
    $ python main.py  # remember to set the TUSHARE_TOKEN env‑var first
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta # Added for date calculation

import numpy as np
# Workaround for pandas_ta NaN import issue
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
import pandas as pd
import pandas_ta as ta  # technical‑analysis helpers
import tushare as ts
import pyarrow # Added for Parquet
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# ────────────────────────────────────────────────────────────────────────────────
# 1. Config
# ────────────────────────────────────────────────────────────────────────────────
TOKEN = os.getenv("TUSHARE_API_KEY")  # Changed to TUSHARE_API_KEY
if not TOKEN:
    sys.exit("❌  Please set the TUSHARE_API_KEY environment variable first!")

SYMBOL = "AAPL.O"          # Changed to Apple Inc. (NASDAQ)
# Calculate start date for 5 years of data
end_date = datetime.now()
start_date_dt = end_date - timedelta(days=5*365)
START_DATE = start_date_dt.strftime("%Y%m%d")

TEST_SIZE = 0.2                # 20 % of data kept for hold‑out test
UP_THRESHOLD = 0.002           # +0.2 %.

# XGBoost hyper‑params – tweak away
XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
)

# ────────────────────────────────────────────────────────────────────────────────
# 2. Data download
# ────────────────────────────────────────────────────────────────────────────────
CACHE_FILE = Path("data_cache.parquet")

ts.set_token(TOKEN)
pro = ts.pro_api()

if CACHE_FILE.exists():
    print(f"♻️  Loading data from cache: {CACHE_FILE} …")
    df = pd.read_parquet(CACHE_FILE)
else:
    print(f"📥  Fetching daily bars for {SYMBOL} …")
    df = pro.daily(ts_code=SYMBOL, start_date=START_DATE)
    if df.empty:
        sys.exit("No data returned – check symbol and date range.")
    print(f"💾  Saving data to cache: {CACHE_FILE} …")
    df.to_parquet(CACHE_FILE)

df.sort_values("trade_date", inplace=True)  # chronological order

# ────────────────────────────────────────────────────────────────────────────────
# 3. Feature engineering
# ────────────────────────────────────────────────────────────────────────────────
print("🛠️   Engineering features …")

# Simple Moving Averages & their day‑to‑day diffs
for win in (5, 10, 20):
    df[f"SMA{win}"] = ta.sma(df["close"], length=win)
    df[f"SMA{win}_diff"] = df[f"SMA{win}"].pct_change()

# Relative Strength Index
df["RSI_14"] = ta.rsi(df["close"], length=14)

# MACD histogram
macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
df["MACD_hist"] = macd["MACDh_12_26_9"]

# Volume‑based signal
df["Volume_SMA5"] = ta.sma(df["vol"], length=5)
df["Volume_SMA5_ratio"] = df["vol"] / df["Volume_SMA5"]

# Label: next‑day return
print("🏷️   Labelling targets …")
df["future_return"] = df["close"].shift(-1) / df["close"] - 1.0

df["target"] = (df["future_return"] >= UP_THRESHOLD).astype(int)

# Keep only the necessary columns & drop NaNs from rolling calcs
FEATURES = [
    "SMA20",
    "SMA5_diff",
    "SMA10_diff",
    "SMA20_diff",
    "RSI_14",
    "MACD_hist",
    "Volume_SMA5_ratio",
    "vol",
]

df = df[FEATURES + ["target"]].dropna().reset_index(drop=True)

# ────────────────────────────────────────────────────────────────────────────────
# 4. Train‑test split (time‑series style)
# ────────────────────────────────────────────────────────────────────────────────
print("✂️   Splitting train/test chronologically …")
split_idx = int(len(df) * (1 - TEST_SIZE))
X_train, X_test = df.iloc[:split_idx][FEATURES], df.iloc[split_idx:][FEATURES]
y_train, y_test = df.iloc[:split_idx]["target"], df.iloc[split_idx:]["target"]

# ────────────────────────────────────────────────────────────────────────────────
# 5. Model fit
# ────────────────────────────────────────────────────────────────────────────────
print("🚂  Fitting XGBoost …")
model = XGBClassifier(**XGB_PARAMS)
model.fit(X_train, y_train)

# ────────────────────────────────────────────────────────────────────────────────
# 6. Evaluation
# ────────────────────────────────────────────────────────────────────────────────
print("🔍  Evaluating …")
prob = model.predict_proba(X_test)[:, 1]
y_pred = (prob >= 0.5).astype(int)

print("\nCLF report (Up ≥ +0.2 %) vs. (Down/Flat):\n")
print(classification_report(y_test, y_pred, target_names=["Down/Flat", "Up ≥0.2%"], digits=2)) # Adjusted digits for screenshot consistency
print("Raw accuracy:", accuracy_score(y_test, y_pred).round(2)) # Adjusted digits for screenshot consistency

# Also print the report on the train set for comparison
print("\nCLF report on TRAIN set (Up ≥ +0.2 %) vs. (Down/Flat):\n")
y_predict_train = model.predict(X_train)
print(classification_report(y_train, y_predict_train, target_names=["Down/Flat", "Up ≥0.2%"], digits=2)) # Adjusted digits for screenshot consistency
print("Raw accuracy on TRAIN set:", accuracy_score(y_train, y_predict_train).round(2)) # Adjusted digits for screenshot consistency

# ────────────────────────────────────────────────────────────────────────────────
# 7. Save the model if you like
# ────────────────────────────────────────────────────────────────────────────────
# from joblib import dump; dump(model, "xgb_model.joblib")