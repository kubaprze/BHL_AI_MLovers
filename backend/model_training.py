import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import logging

# Suppress statsmodels warnings
logging.getLogger('statsmodels').setLevel(logging.ERROR)

print("=" * 80)
print("TIME SERIES MODEL TRAINING (ARIMA/SARIMA)")
print("=" * 80)




df = None
csv_path = "../data/company_growth_detailed.csv"

df = pd.read_csv(csv_path)


df['employees'] = pd.to_numeric(df['employees'], errors='coerce')
df = df.dropna(subset=['employees'])
df['employees'] = df['employees'].astype(int)

print(f"📊 Data shape: {len(df)} months")
print(f"   Start: {df['employees'].iloc[0]} employees")
print(f"   End: {df['employees'].iloc[-1]} employees")

# ============ PREPARE TIME SERIES ============

ts_data = df['employees'].values

print(f"\n📈 Time Series Data:")
print(f"   Length: {len(ts_data)}")
print(f"   Mean: {ts_data.mean():.0f}")
print(f"   Std: {ts_data.std():.0f}")

# Train/Test split
train_size = int(0.8 * len(ts_data))
train, test = ts_data[:train_size], ts_data[train_size:]

print(f"\n📊 Train/Test Split:")
print(f"   Training: {len(train)} samples")
print(f"   Testing: {len(test)} samples")

# ============ TEST ARIMA MODELS ============

print(f"\n🤖 Testing ARIMA Models...")

models_to_test = [
    ((1, 1, 1), (0, 0, 0, 0)),    # ARIMA(1,1,1)
    ((2, 1, 1), (0, 0, 0, 0)),    # ARIMA(2,1,1)
    ((1, 1, 2), (0, 0, 0, 0)),    # ARIMA(1,1,2)
    ((1, 1, 1), (1, 0, 1, 12)),   # SARIMA(1,1,1)(1,0,1,12)
    ((2, 1, 2), (1, 0, 1, 12)),   # SARIMA(2,1,2)(1,0,1,12)
]

best_model = None
best_mae = float('inf')
best_config = None
best_model_fit = None

for order, seasonal_order in models_to_test:
    try:
        # Fit model
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            disp=False
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_model = model.fit(disp=False)
        
        # Make test predictions
        forecast_output = fitted_model.get_forecast(steps=len(test))
        predictions = np.array(forecast_output.predicted_mean)
        
        # Calculate MAE
        mae = mean_absolute_error(test, predictions)
        
        model_name = f"SARIMA{order}{seasonal_order}" if seasonal_order[3] > 0 else f"ARIMA{order}"
        print(f"   ✅ {model_name}: MAE = {mae:.2f}")
        
        if mae < best_mae:
            best_mae = mae
            best_model = fitted_model
            best_model_fit = model
            best_config = (order, seasonal_order)
    
    except Exception as e:
        model_name = f"SARIMA{order}{seasonal_order}" if seasonal_order[3] > 0 else f"ARIMA{order}"
        print(f"   ⚠️  {model_name}: {str(e)[:60]}")

if best_model is None:
    print("\n❌ No models trained successfully!")
    exit(1)

print(f"\n✅ Best Model: ARIMA{best_config[0]} or SARIMA{best_config[0]}{best_config[1]}")
print(f"   Test MAE: {best_mae:.2f} employees")

# ============ RETRAIN ON FULL DATA ============

print(f"\n🔄 Retraining on full dataset...")

final_model_fit = SARIMAX(
    ts_data,
    order=best_config[0],
    seasonal_order=best_config[1],
    enforce_stationarity=False,
    enforce_invertibility=False,
    disp=False
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    final_model = final_model_fit.fit(disp=False)

print(f"✅ Model retrained on {len(ts_data)} samples")

# ============ SAVE MODELS ============

models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"📁 Created directory: {models_dir}")

# Save SARIMA model
sarima_path = os.path.join(models_dir, "sarima_model.pkl")
with open(sarima_path, 'wb') as f:
    pickle.dump(final_model, f)
print(f"✅ Saved SARIMA model: {sarima_path}")

# Save training data
data_path = os.path.join(models_dir, "training_data.csv")
df.to_csv(data_path, index=False)
print(f"✅ Saved training data: {data_path}")

# Save metadata
metadata_ts = {
    "model_type": "SARIMA",
    "order": list(best_config[0]),
    "seasonal_order": list(best_config[1]),
    "test_mae": float(best_mae),
    "train_size": int(len(train)),
    "test_size": int(len(test)),
    "total_samples": int(len(ts_data)),
    "trained_date": datetime.now().isoformat(),
    "source_file": csv_path,
    "model_name": f"SARIMA{best_config[0]}{best_config[1]}"
}

metadata_path = os.path.join(models_dir, "sarima_metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata_ts, f, indent=2)
print(f"✅ Saved metadata: {metadata_path}")

print("\n" + "=" * 80)
print("✅ TIME SERIES MODEL TRAINING COMPLETE!")
print("=" * 80)
print(f"\nModel Summary:")
print(f"  - Type: SARIMA")
print(f"  - Order (p,d,q): {best_config[0]}")
print(f"  - Seasonal (P,D,Q,s): {best_config[1]}")
print(f"  - Test MAE: {best_mae:.2f} employees")
print(f"  - Training Samples: {len(train)}")
print(f"  - Test Samples: {len(test)}")