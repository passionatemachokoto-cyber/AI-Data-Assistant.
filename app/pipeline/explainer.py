import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ===============================
# LOAD DATA
# ===============================
csv_path = "data/clean_house_prices.csv"
print(f"Loading data from: {csv_path}")

df = pd.read_csv(csv_path)
print("Columns:", df.columns)

# ===============================
# DEFINE TARGET
# ===============================
target = "price"

if target not in df.columns:
    raise ValueError("Target column 'price' not found in dataset")

X = df.drop(columns=[target])
y = df[target]

# ===============================
# TRAIN MODEL
# ===============================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)

# ===============================
# FEATURE IMPORTANCE
# ===============================
importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance_df.head(10))

# ===============================
# PLOT
# ===============================
top_features = feature_importance_df.head(10)

plt.figure()
plt.barh(top_features["Feature"], top_features["Importance"])
plt.gca().invert_yaxis()
plt.title("Top 10 Features Affecting House Price")
plt.xlabel("Importance")
plt.show()
