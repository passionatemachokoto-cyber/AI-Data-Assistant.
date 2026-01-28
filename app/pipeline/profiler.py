import os
import pandas as pd

# Find project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Find CSV
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError("No CSV file found in data folder")

csv_path = os.path.join(DATA_DIR, csv_files[0])
print(f"\nLoaded file: {csv_files[0]}\n")

# Load data
df = pd.read_csv(csv_path)

# ---- PROFILING ----
print("📊 DATA PROFILE\n")

print("Columns:")
print(df.columns.tolist(), "\n")

print("Data types:")
print(df.dtypes, "\n")

print("Missing values:")
print(df.isnull().sum(), "\n")

print("Basic statistics:")
print(df.describe(include="all"))
