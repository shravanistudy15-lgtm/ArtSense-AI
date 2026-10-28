from apis.met_api import (
    search_met_artworks,
    met_results_to_dataframe
)

import os


# ==================================================
# CREATE OUTPUT FOLDER
# ==================================================

os.makedirs(
    "api_data",
    exist_ok=True
)


# ==================================================
# FETCH ARTWORK DATA
# ==================================================

print(
    "\nFetching artwork data from The Met API..."
)


artworks = search_met_artworks(
    query="painting",
    limit=50
)


# ==================================================
# CONVERT TO DATAFRAME
# ==================================================

df = met_results_to_dataframe(
    artworks
)


# ==================================================
# SAVE CSV
# ==================================================

output_path = (
    "api_data/met_artworks.csv"
)


df.to_csv(
    output_path,
    index=False
)


print("\nMet API data saved successfully!")

print("File:", output_path)

print("Shape:", df.shape)

print("\nColumns:")

print(df.columns.tolist())

print("\nFirst 5 rows:")

print(df.head())