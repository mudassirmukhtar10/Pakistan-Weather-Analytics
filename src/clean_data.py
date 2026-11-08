import glob
import pandas as pd

# 1. Data folder se saari CSV files uthao
all_files = glob.glob("Data/*.csv")
df_list = []

for filename in all_files:
  df = pd.read_csv(filename)

  # File name se city ka naam nikal kar column bana lo
  city_name = (
      filename.split("/")[-1]
      .split("\\")[-1]
      .replace(".csv", "")
      .capitalize()
  )
  df["city"] = city_name

  df_list.append(df)

# Sab cities ka data ek dataframe mein combine karo
final_df = pd.concat(df_list, ignore_index=True)

print("Original total rows:", len(final_df))

# 2. Missing values check karo
print("Missing values:\n", final_df.isnull().sum())

# 3. Duplicate records remove karo
final_df = final_df.drop_duplicates()

# 4. Date column ko proper datetime format mein convert karo
if "time" in final_df.columns:
  final_df["time"] = pd.to_datetime(final_df["time"])

# 5. Cleaned data ko nayi CSV file mein save karo
final_df.to_csv("Data/cleaned_weather.csv", index=False)
print("Cleaning complete! 'cleaned_weather.csv' file Data is done in folder .")