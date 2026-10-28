import requests
import pandas as pd
import os
from dotenv import load_dotenv


# --------------------------------------------------
# SMITHSONIAN OPEN ACCESS API
# --------------------------------------------------

print("=" * 60)
print("FETCHING ARTWORK DATA FROM SMITHSONIAN OPEN ACCESS")
print("=" * 60)


# --------------------------------------------------
# STEP 1: LOAD API KEY
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("SMITHSONIAN_API_KEY")

if not API_KEY:
    raise ValueError(
        "SMITHSONIAN_API_KEY not found in .env file."
    )


# --------------------------------------------------
# STEP 2: CREATE OUTPUT FOLDER
# --------------------------------------------------

os.makedirs("api_data", exist_ok=True)


# --------------------------------------------------
# STEP 3: API URL
# --------------------------------------------------

API_URL = (
    "https://api.si.edu/openaccess/api/v1.0/search"
)


# --------------------------------------------------
# STEP 4: SEARCH PARAMETERS
# --------------------------------------------------

params = {
    "q": "painting",
    "rows": 50,
    "start": 0,
    "api_key": API_KEY
}


# --------------------------------------------------
# STEP 5: FETCH DATA
# --------------------------------------------------

print("\nSearching Smithsonian API for: painting")

try:

    response = requests.get(
        API_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    response_data = data.get("response", {})

    artworks = response_data.get("rows", [])

    print(f"Found {len(artworks)} artwork records.")

except Exception as e:

    print("\nError while fetching Smithsonian API data:")
    print(e)

    artworks = []


# --------------------------------------------------
# STEP 6: HELPER FUNCTION
# --------------------------------------------------

def get_value_from_indexed_data(
    indexed_data,
    key
):
    """
    Safely get a value from indexedStructured data.
    """

    values = indexed_data.get(key, [])

    if isinstance(values, list) and len(values) > 0:
        return ", ".join(
            str(value)
            for value in values
        )

    return ""


# --------------------------------------------------
# STEP 7: CONVERT TO STANDARD FORMAT
# --------------------------------------------------

records = []

for artwork in artworks:

    content = artwork.get("content", {})

    indexed_data = content.get(
        "indexedStructured",
        {}
    )

    descriptive_data = content.get(
        "descriptiveNonRepeating",
        {}
    )


    # ----------------------------------------------
    # TITLE
    # ----------------------------------------------

    title = content.get(
        "title",
        "Untitled"
    )


    # ----------------------------------------------
    # ARTIST
    # ----------------------------------------------

    artist = get_value_from_indexed_data(
        indexed_data,
        "name"
    )


    # ----------------------------------------------
    # DATE
    # ----------------------------------------------

    year_creation = get_value_from_indexed_data(
        indexed_data,
        "date"
    )


    # ----------------------------------------------
    # TYPE
    # ----------------------------------------------

    artwork_type = get_value_from_indexed_data(
        indexed_data,
        "object_type"
    )


    # ----------------------------------------------
    # PLACE / COUNTRY
    # ----------------------------------------------

    country = get_value_from_indexed_data(
        indexed_data,
        "place"
    )


    # ----------------------------------------------
    # DESCRIPTION
    # ----------------------------------------------

    description = ""

    notes = content.get(
        "freetext",
        {}
    )

    if isinstance(notes, dict):

        physical_description = notes.get(
            "physicalDescription",
            []
        )

        if (
            isinstance(
                physical_description,
                list
            )
            and len(physical_description) > 0
        ):

            first_description = physical_description[0]

            if isinstance(
                first_description,
                dict
            ):
                description = first_description.get(
                    "content",
                    ""
                )


    # ----------------------------------------------
    # IMAGE URL
    # ----------------------------------------------

    image_url = ""

    online_media = descriptive_data.get(
        "online_media",
        {}
    )

    media_items = online_media.get(
        "media",
        []
    )

    if (
        isinstance(media_items, list)
        and len(media_items) > 0
    ):

        first_media = media_items[0]

        if isinstance(first_media, dict):

            image_url = first_media.get(
                "content",
                ""
            )


    # ----------------------------------------------
    # CREATE STANDARD RECORD
    # ----------------------------------------------

    record = {

        "source": "Smithsonian Open Access",

        "api_id": artwork.get(
            "id",
            ""
        ),

        "title": title,

        "artist": artist,

        "yearCreation": year_creation,

        "period": "",

        "movement": "",

        "medium": artwork_type,

        "department": content.get(
            "unitCode",
            ""
        ),

        "country": country,

        "image_url": image_url,

        "description": description
    }

    records.append(record)


# --------------------------------------------------
# STEP 8: CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(records)


# --------------------------------------------------
# STEP 9: SAVE CSV
# --------------------------------------------------

OUTPUT_FILE = (
    "api_data/smithsonian_artworks.csv"
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# STEP 10: SHOW RESULTS
# --------------------------------------------------

print("\nSmithsonian API data saved successfully!")

print("\nFile:")
print(OUTPUT_FILE)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


print("\n" + "=" * 60)
print("SMITHSONIAN API COMPLETED")
print("=" * 60)