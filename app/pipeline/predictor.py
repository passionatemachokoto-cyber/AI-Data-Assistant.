import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Load cleaned data
data_path = os.path.join("data", "clean_house_prices.csv")
df = pd.read_csv(data_path)

# Detect target safely
if "Price" in df.columns:
    target = "Price"
elif "price" in df.columns:
    target = "price"
else:
    raise ValueError("Target column not found")

X = df.drop(columns=[target])
y = df[target]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/house_price_model.pkl")

# Example prediction (first row)
sample = X.iloc[[0]]
prediction = model.predict(sample)

print("Predicted house price:", prediction[0])
