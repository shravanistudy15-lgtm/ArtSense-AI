import requests
import pandas as pd
import os


# --------------------------------------------------
# VICTORIA AND ALBERT MUSEUM API
# --------------------------------------------------

print("=" * 60)
print("FETCHING ARTWORK DATA FROM V&A MUSEUM")
print("=" * 60)


# --------------------------------------------------
# STEP 1: CREATE OUTPUT FOLDER
# --------------------------------------------------

os.makedirs("api_data", exist_ok=True)


# --------------------------------------------------
# STEP 2: API URL
# --------------------------------------------------

API_URL = "https://api.vam.ac.uk/v2/objects/search"


# --------------------------------------------------
# STEP 3: PARAMETERS
# --------------------------------------------------

params = {
    "q": "painting",
    "page_size": 50
}


# --------------------------------------------------
# STEP 4: FETCH DATA
# --------------------------------------------------

print("\nSearching V&A Museum API for: painting")

try:

    response = requests.get(
        API_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    artworks = data.get("records", [])

    print(f"Found {len(artworks)} artwork records.")

except Exception as e:

    print("\nError while fetching V&A API data:")
    print(e)

    artworks = []


# --------------------------------------------------
# STEP 5: CONVERT DATA TO STANDARD FORMAT
# --------------------------------------------------

records = []

for artwork in artworks:

    # ----------------------------------------------
    # ARTIST / MAKER
    # ----------------------------------------------

    primary_maker = artwork.get("_primaryMaker", {})

    if isinstance(primary_maker, dict):
        artist = primary_maker.get("name", "Unknown")
    else:
        artist = "Unknown"


    # ----------------------------------------------
    # DATE / YEAR
    # ----------------------------------------------

    primary_date = artwork.get("_primaryDate", "")

    year_creation = str(primary_date)


    # ----------------------------------------------
    # IMAGE URL
    # ----------------------------------------------

    image_url = ""

    images = artwork.get("_images", [])

    if isinstance(images, list) and len(images) > 0:

        first_image = images[0]

        if isinstance(first_image, dict):
            image_url = first_image.get("thumbnail", "")


    # ----------------------------------------------
    # DESCRIPTION
    # ----------------------------------------------

    description = artwork.get(
        "summaryDescription",
        ""
    )


    # ----------------------------------------------
    # STYLE
    # ----------------------------------------------

    style = artwork.get(
        "_sampleStyle",
        ""
    )


    # ----------------------------------------------
    # MATERIAL
    # ----------------------------------------------

    material = artwork.get(
        "_sampleMaterial",
        ""
    )


    # ----------------------------------------------
    # TECHNIQUE
    # ----------------------------------------------

    technique = artwork.get(
        "_sampleTechnique",
        ""
    )


    # ----------------------------------------------
    # CREATE STANDARD RECORD
    # ----------------------------------------------

    record = {

        "source": "Victoria and Albert Museum",

        "api_id": artwork.get(
            "systemNumber",
            ""
        ),

        "title": artwork.get(
            "_primaryTitle",
            "Untitled"
        ),

        "artist": artist,

        "yearCreation": year_creation,

        "period": artwork.get(
            "_primaryDate",
            ""
        ),

        "movement": style,

        "medium": material,

        "technique": technique,

        "department": artwork.get(
            "objectType",
            ""
        ),

        "country": artwork.get(
            "_primaryPlace",
            ""
        ),

        "image_url": image_url,

        "description": description
    }

    records.append(record)


# --------------------------------------------------
# STEP 6: CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(records)


# --------------------------------------------------
# STEP 7: SAVE CSV
# --------------------------------------------------

OUTPUT_FILE = "api_data/va_artworks.csv"

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# STEP 8: SHOW RESULTS
# --------------------------------------------------

print("\nV&A API data saved successfully!")

print("\nFile:")
print(OUTPUT_FILE)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


print("\n" + "=" * 60)
print("V&A MUSEUM API COMPLETED")
print("=" * 60)