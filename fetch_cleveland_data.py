import requests
import pandas as pd
import os


# --------------------------------------------------
# CLEVELAND MUSEUM OF ART API
# --------------------------------------------------

print("=" * 60)
print("FETCHING ARTWORK DATA FROM CLEVELAND MUSEUM OF ART")
print("=" * 60)


# --------------------------------------------------
# STEP 1: CREATE OUTPUT FOLDER
# --------------------------------------------------

os.makedirs("api_data", exist_ok=True)


# --------------------------------------------------
# STEP 2: API URL
# --------------------------------------------------

API_URL = "https://openaccess-api.clevelandart.org/api/artworks/"


# --------------------------------------------------
# STEP 3: PARAMETERS
# --------------------------------------------------

params = {
    "limit": 50,
    "has_image": 1
}


# --------------------------------------------------
# STEP 4: FETCH DATA
# --------------------------------------------------

print("\nFetching artwork data...")

try:

    response = requests.get(
        API_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    artworks = data.get("data", [])

    print(f"Found {len(artworks)} artworks.")

except Exception as e:

    print("\nError while fetching Cleveland API data:")
    print(e)

    artworks = []


# --------------------------------------------------
# STEP 5: CONVERT DATA TO STANDARD FORMAT
# --------------------------------------------------

records = []

for artwork in artworks:

    # Artist / creator
    creators = artwork.get("creators", [])

    if creators:
        artist = creators[0].get("description", "")
    else:
        artist = ""


    # Year / date
    creation_date = artwork.get("creation_date", "")

    if isinstance(creation_date, dict):
        year_creation = creation_date.get("earliest", "")
        period = creation_date.get("text", "")
    else:
        year_creation = ""
        period = str(creation_date)


    # Image URL
    image_data = artwork.get("images", {})

    image_url = ""

    if isinstance(image_data, dict):
        web_image = image_data.get("web", {})

        if isinstance(web_image, dict):
            image_url = web_image.get("url", "")


    # Create standard record
    record = {

        "source": "Cleveland Museum of Art",

        "api_id": artwork.get("id", ""),

        "title": artwork.get("title", ""),

        "artist": artist,

        "yearCreation": year_creation,

        "period": period,

        "movement": artwork.get("style", ""),

        "medium": artwork.get("technique", ""),

        "department": artwork.get("department", ""),

        "country": artwork.get("culture", ""),

        "image_url": image_url,

        "description": artwork.get("description", "")
    }

    records.append(record)


# --------------------------------------------------
# STEP 6: CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(records)


# --------------------------------------------------
# STEP 7: SAVE CSV
# --------------------------------------------------

OUTPUT_FILE = "api_data/cleveland_artworks.csv"

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# STEP 8: SHOW RESULTS
# --------------------------------------------------

print("\nCleveland API data saved successfully!")

print("\nFile:")
print(OUTPUT_FILE)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\n" + "=" * 60)
print("CLEVELAND MUSEUM API COMPLETED")
print("=" * 60)