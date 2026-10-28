import requests
import pandas as pd
import os


# --------------------------------------------------
# ART INSTITUTE OF CHICAGO API
# --------------------------------------------------

print("=" * 60)
print("FETCHING ARTWORK DATA FROM ART INSTITUTE OF CHICAGO")
print("=" * 60)


# --------------------------------------------------
# STEP 1: CREATE OUTPUT FOLDER
# --------------------------------------------------

os.makedirs("api_data", exist_ok=True)


# --------------------------------------------------
# STEP 2: API URL
# --------------------------------------------------

API_URL = "https://api.artic.edu/api/v1/artworks"


# --------------------------------------------------
# STEP 3: PARAMETERS
# --------------------------------------------------

params = {
    "limit": 50,
    "fields": (
        "id,title,artist_display,date_display,"
        "date_start,date_end,medium_display,"
        "artwork_type_title,style_title,"
        "place_of_origin,description,"
        "department_title,image_id"
    )
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

    print("\nError while fetching Chicago API data:")
    print(e)

    artworks = []


# --------------------------------------------------
# STEP 5: CONVERT TO STANDARD FORMAT
# --------------------------------------------------

records = []

for artwork in artworks:

    record = {

        # Source information
        "source": "Art Institute of Chicago",

        # API ID
        "api_id": artwork.get("id", ""),

        # Basic artwork information
        "title": artwork.get("title", ""),

        # Artist
        "artist": artwork.get("artist_display", ""),

        # Year information
        "yearCreation": artwork.get("date_start", ""),

        # Art period / date
        "period": artwork.get("date_display", ""),

        # Style / movement
        "movement": artwork.get("style_title", ""),

        # Medium
        "medium": artwork.get("medium_display", ""),

        # Artwork type
        "artwork_type": artwork.get(
            "artwork_type_title",
            ""
        ),

        # Department
        "department": artwork.get(
            "department_title",
            ""
        ),

        # Country / place of origin
        "country": artwork.get(
            "place_of_origin",
            ""
        ),

        # Image ID
        "image_id": artwork.get(
            "image_id",
            ""
        ),

        # Description
        "description": artwork.get(
            "description",
            ""
        )
    }

    records.append(record)


# --------------------------------------------------
# STEP 6: CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(records)


# --------------------------------------------------
# STEP 7: SAVE CSV
# --------------------------------------------------

OUTPUT_FILE = "api_data/chicago_artworks.csv"

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# STEP 8: SHOW RESULTS
# --------------------------------------------------

print("\nChicago API data saved successfully!")

print("\nFile:")
print(OUTPUT_FILE)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\n" + "=" * 60)
print("ART INSTITUTE OF CHICAGO API COMPLETED")
print("=" * 60)