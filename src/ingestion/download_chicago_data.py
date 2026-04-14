import os
import pandas as pd
import requests

os.makedirs("data/raw", exist_ok=True)

url = (
    "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"
    "?$select=id,date,primary_type,description,location_description,"
    "arrest,domestic,beat,district,ward,community_area,year,latitude,longitude"
    "&$where=year>=2023 AND latitude IS NOT NULL AND longitude IS NOT NULL"
    "&$limit=50000"
)

output_path = "data/raw/chicago_crimes_2023_2026.csv"

response = requests.get(url, timeout=60)
response.raise_for_status()

with open(output_path, "wb") as f:
    f.write(response.content)

df = pd.read_csv(output_path)
print("Saved file to:", output_path)
print("Shape:", df.shape)
print(df.head())