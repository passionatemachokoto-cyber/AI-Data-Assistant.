# app/pipeline/cleaner.py

import pandas as pd
from pathlib import Path

# Locate data folder
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

# Load CSV
csv_files = list(DATA_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

df = pd.read_csv(csv_files[0])
print(f"Loaded file: {csv_files[0].name}")

# -------------------------
# CLEANING STEPS
# -------------------------

# 1. Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
)

# 2. Drop duplicate rows
df = df.drop_duplicates()

# 3. Handle missing values
for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna("unknown")

# -------------------------
# SAVE CLEAN DATA
# -------------------------

clean_path = DATA_DIR / "clean_house_prices.csv"
df.to_csv(clean_path, index=False)

print("Cleaning complete.")
print(f"Clean data saved to: {clean_path}")
print("Final shape:", df.shape)
