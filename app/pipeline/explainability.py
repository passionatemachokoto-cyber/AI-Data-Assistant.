import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os

# Load cleaned data
data_path = os.path.join("data", "clean_house_prices.csv")
df = pd.read_csv(data_path)

# Fix target column name safely
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

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

# Save to CSV
os.makedirs("outputs", exist_ok=True)
importances.to_csv("outputs/feature_importance.csv")

# Plot
plt.figure()
importances.head(10).plot(kind="bar")
plt.title("Top 10 Most Important Features")
plt.tight_layout()
plt.show()
