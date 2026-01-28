# app/pipeline/modeler.py

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# -------------------------
# LOAD FEATURE DATA
# -------------------------

feature_file = DATA_DIR / "feature_house_prices.csv"
if not feature_file.exists():
    raise FileNotFoundError("Run feature_engineer first")

df = pd.read_csv(feature_file)
print("Loaded feature dataset")

# -------------------------
# SELECT FEATURES
# -------------------------

target = "price"

X = df.drop(columns=[target])
y = df[target]

# Keep only numeric features
X = X.select_dtypes(include="number")

# -------------------------
# TRAIN / TEST SPLIT
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# TRAIN MODEL
# -------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------
# EVALUATE
# -------------------------

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print(f"Model MAE: {mae:,.2f}")

# -------------------------
# SAVE MODEL
# -------------------------

model_path = MODEL_DIR / "house_price_model.pkl"
joblib.dump(model, model_path)

print(f"Model saved to {model_path}")
