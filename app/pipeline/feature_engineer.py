# app/pipeline/feature_engineer.py

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

# Load cleaned data
clean_file = DATA_DIR / "clean_house_prices.csv"
if not clean_file.exists():
    raise FileNotFoundError(f"{clean_file} not found")

df = pd.read_csv(clean_file)
print("Loaded cleaned dataset")

# -------------------------
# FEATURE ENGINEERING
# -------------------------

# Total house area
if "area_of_the_houseexcluding_basement" in df.columns and "area_of_the_basement" in df.columns:
    df["total_house_area"] = (
        df["area_of_the_houseexcluding_basement"] +
        df["area_of_the_basement"]
    )

# House age
if "built_year" in df.columns:
    df["house_age"] = 2025 - df["built_year"]

# Renovated or not
if "renovation_year" in df.columns:
    df["is_renovated"] = df["renovation_year"].apply(lambda x: 1 if x > 0 else 0)

# -------------------------
# SAVE FEATURE DATA
# -------------------------

feature_path = DATA_DIR / "feature_house_prices.csv"
df.to_csv(feature_path, index=False)

print("Feature engineering complete.")
print(f"Feature data saved to: {feature_path}")
print("Final shape:", df.shape)
