import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

# Output path in project root
ROOT_DIR = Path(__file__).parent.parent
OUTPUT_CSV = ROOT_DIR / "weather_data.csv"

# MySQL connection details (Port 3307 for weather_db)
DB_USER = "root"
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_PORT = "3307"
DB_NAME = "weather_db"
TABLE_NAME = "weather_data"

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

query = f"SELECT * FROM {TABLE_NAME}"
print(f"Fetching records from MySQL {DB_NAME}.{TABLE_NAME} on port {DB_PORT}...")
df = pd.read_sql(query, engine)

df.to_csv(OUTPUT_CSV, index=False)
print(f"Mubarak ho! {len(df)} records saved to {OUTPUT_CSV}")
