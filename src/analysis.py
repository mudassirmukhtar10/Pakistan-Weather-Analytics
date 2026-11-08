import pandas as pd
from sqlalchemy import create_engine

# 1. Cleaned CSV file load karo
df = pd.read_csv("Data/cleaned_weather.csv")

# 2. MySQL Connection Setup (Port 3307 ke sath)
db_user = "root"
db_password = ""  # Password khali hai
db_host = "localhost"
db_port = "3307"  # Port 3307 jo humne XAMPP mein set kiya hai
db_name = "weather_db"

engine = create_engine(
    f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
)

# 3. DataFrame ko MySQL table mein save karo
df.to_sql("weather_data", con=engine, if_exists="replace", index=False)
print("Data successfully saved in MySQL database !")

# 4. Check karne ke liye ek query run karte hain
query = "SELECT city, COUNT(*) as total_records FROM weather_data GROUP BY city"
result = pd.read_sql(query, con=engine)
print("\n--- MySQL Weather Insights ---")
print(result)