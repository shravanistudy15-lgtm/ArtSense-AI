import pandas as pd
import os
import re


# ============================================================
# ARTSENSE AI
# FINAL DATA FUSION
# ORIGINAL DATASET + API DATASET
# ============================================================

print("=" * 70)
print("ARTSENSE AI - FINAL DATA FUSION")
print("ORIGINAL DATASET + API DATASET")
print("=" * 70)


# ============================================================
# STEP 1: FILE PATHS
# ============================================================

ORIGINAL_PATH = "data/artworks.csv.csv"

API_PATH = "fused_data/artsense_fused_dataset.csv"

OUTPUT_FOLDER = "fused_data"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# STEP 2: CHECK FILES
# ============================================================

print("\nChecking files...")


if not os.path.exists(ORIGINAL_PATH):

    raise FileNotFoundError(
        f"Original dataset not found: {ORIGINAL_PATH}"
    )


if not os.path.exists(API_PATH):

    raise FileNotFoundError(
        f"API dataset not found: {API_PATH}"
    )


print("Original dataset found.")
print("API dataset found.")


# ============================================================
# STEP 3: LOAD ORIGINAL DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING ORIGINAL DATASET")
print("=" * 70)


original_df = pd.read_csv(
    ORIGINAL_PATH
)


print(
    "Original dataset shape:",
    original_df.shape
)


print(
    "\nOriginal columns:"
)

print(
    original_df.columns.tolist()
)


# ============================================================
# STEP 4: LOAD API DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING API DATASET")
print("=" * 70)


api_df = pd.read_csv(
    API_PATH
)


print(
    "API dataset shape:",
    api_df.shape
)


print(
    "\nAPI columns:"
)

print(
    api_df.columns.tolist()
)


# ============================================================
# STEP 5: REMOVE UNNECESSARY ORIGINAL INDEX
# ============================================================

if "Unnamed: 0" in original_df.columns:

    original_df = original_df.drop(
        columns=["Unnamed: 0"]
    )


# ============================================================
# STEP 6: ADD RECORD SOURCE
# ============================================================

original_df["record_source"] = "Original Dataset"

api_df["record_source"] = "API Dataset"


# ============================================================
# STEP 7: TEXT NORMALIZATION FUNCTION
# ============================================================

def normalize_text(value):

    if pd.isna(value):

        return ""

    value = str(value).lower()

    # Remove punctuation
    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value
    )

    # Remove extra spaces
    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# ============================================================
# STEP 8: CREATE MATCHING COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("CREATING MATCHING KEYS")
print("=" * 70)


# Original dataset
original_df["title_normalized"] = (
    original_df["title"]
    .apply(normalize_text)
)


original_df["artist_normalized"] = (
    original_df["artist"]
    .apply(normalize_text)
)


# API dataset
api_df["title_normalized"] = (
    api_df["title"]
    .apply(normalize_text)
)


api_df["artist_normalized"] = (
    api_df["artist"]
    .apply(normalize_text)
)


# ============================================================
# STEP 9: CREATE COMPOSITE KEY
# ============================================================

original_df["match_key"] = (

    original_df["title_normalized"]

    + " | "

    + original_df["artist_normalized"]

)


api_df["match_key"] = (

    api_df["title_normalized"]

    + " | "

    + api_df["artist_normalized"]

)


# ============================================================
# STEP 10: CHECK DUPLICATE KEYS
# ============================================================

print("\nChecking duplicate matching keys...")


original_duplicate_keys = (

    original_df[
        original_df["match_key"].duplicated(
            keep=False
        )
    ]

)


api_duplicate_keys = (

    api_df[
        api_df["match_key"].duplicated(
            keep=False
        )
    ]

)


print(
    "Duplicate keys in original dataset:",
    len(original_duplicate_keys)
)


print(
    "Duplicate keys in API dataset:",
    len(api_duplicate_keys)
)


# ============================================================
# STEP 11: FIND MATCHING RECORDS
# ============================================================

print("\n" + "=" * 70)
print("MATCHING ANALYSIS")
print("=" * 70)


original_keys = set(
    original_df["match_key"]
)


api_keys = set(
    api_df["match_key"]
)


matching_keys = (

    original_keys
    .intersection(api_keys)

)


original_only_keys = (

    original_keys
    - api_keys

)


api_only_keys = (

    api_keys
    - original_keys

)


print(
    "\nUnique matching keys:",
    len(matching_keys)
)


print(
    "Original-only keys:",
    len(original_only_keys)
)


print(
    "API-only keys:",
    len(api_only_keys)
)


# ============================================================
# STEP 12: MATCHING RECORD COUNTS
# ============================================================

matching_original_records = (

    original_df[
        original_df["match_key"].isin(
            matching_keys
        )
    ]

)


matching_api_records = (

    api_df[
        api_df["match_key"].isin(
            matching_keys
        )
    ]

)


print(
    "\nMatching original records:",
    len(matching_original_records)
)


print(
    "Matching API records:",
    len(matching_api_records)
)


# ============================================================
# STEP 13: UNMATCHED RECORDS
# ============================================================

original_unmatched = original_df[
    ~original_df["match_key"].isin(matching_keys)
].copy()


api_unmatched = api_df[
    ~api_df["match_key"].isin(matching_keys)
].copy()


api_unmatched = (

    api_df[
        ~api_df["match_key"].isin(
            matching_keys
        )
    ]

)


print(
    "\nUnmatched original records:",
    len(original_unmatched)
)


print(
    "Unmatched API records:",
    len(api_unmatched)
)


# ============================================================
# STEP 14: CONFLICT CHECK
# ============================================================

print("\n" + "=" * 70)
print("CONFLICT ANALYSIS")
print("=" * 70)


# Merge matching records only
matched_comparison = pd.merge(

    matching_original_records,

    matching_api_records,

    on="match_key",

    how="inner",

    suffixes=(
        "_original",
        "_api"
    )

)


print(
    "Matched comparison records:",
    len(matched_comparison)
)


# ============================================================
# PRICE CONFLICT CHECK
# ============================================================

price_conflicts = 0


if "price_original" in matched_comparison.columns:

    if len(matched_comparison) > 0:

        price_conflicts = (
            matched_comparison[
                "price_original"
            ].notna()
        ).sum()


print(
    "Potential price-related records:",
    price_conflicts
)


# ============================================================
# STEP 15: CREATE FUSED DATASET
# ============================================================

print("\n" + "=" * 70)
print("CREATING FINAL FUSED DATASET")
print("=" * 70)


# ------------------------------------------------------------
# Standardize column names for original dataset
# ------------------------------------------------------------

original_fusion = original_df.copy()


# Original columns
original_fusion["date"] = (

    original_fusion["yearCreation"]

    if "yearCreation" in original_fusion.columns

    else "Unknown"

)


original_fusion["medium"] = "Unknown"

original_fusion["department"] = "Unknown"

original_fusion["classification"] = "Unknown"

original_fusion["culture"] = "Unknown"

original_fusion["description"] = "Unknown"

original_fusion["image_url"] = "Unknown"

original_fusion["object_id"] = "Original_" + (

    original_fusion.index.astype(str)

)


original_fusion["source"] = "Original Dataset"


# ------------------------------------------------------------
# Standard columns
# ------------------------------------------------------------

standard_columns = [

    "title",
    "artist",
    "date",
    "medium",
    "department",
    "classification",
    "culture",
    "description",
    "image_url",
    "object_id",
    "source",
    "record_source",
    "price",
    "signed",
    "condition",
    "period",
    "movement"

]


# Add missing original columns
for column in standard_columns:

    if column not in original_fusion.columns:

        original_fusion[column] = "Unknown"


# ------------------------------------------------------------
# API dataset
# ------------------------------------------------------------

api_fusion = api_df.copy()


# API doesn't have original-specific columns
api_specific_columns = [

    "price",
    "signed",
    "condition",
    "period",
    "movement"

]


for column in api_specific_columns:

    if column not in api_fusion.columns:

        api_fusion[column] = "Unknown"


# Make sure standard columns exist
for column in standard_columns:

    if column not in api_fusion.columns:

        api_fusion[column] = "Unknown"


# ------------------------------------------------------------
# Keep same columns
# ------------------------------------------------------------

original_fusion = original_fusion[
    standard_columns
]


api_fusion = api_fusion[
    standard_columns
]


# ============================================================
# STEP 16: CONCATENATE ORIGINAL + API
# ============================================================

print("\nCombining original and API records...")


final_fused_df = pd.concat(

    [
        original_fusion,
        api_fusion
    ],

    ignore_index=True,

    sort=False

)


print(
    "Combined shape before final cleaning:",
    final_fused_df.shape
)


# ============================================================
# STEP 17: CLEAN FINAL DATASET
# ============================================================

print("\nCleaning final fused dataset...")


# Replace missing values
final_fused_df = final_fused_df.fillna(
    "Unknown"
)


# Convert text columns
for column in final_fused_df.columns:

    final_fused_df[column] = (

        final_fused_df[column]

        .astype(str)

        .str.strip()

    )


# ============================================================
# STEP 18: FINAL DUPLICATE CHECK
# ============================================================

duplicate_count = (

    final_fused_df
    .duplicated()
    .sum()

)


print(
    "Exact duplicate records:",
    duplicate_count
)


# ============================================================
# STEP 19: SAVE FINAL CSV
# ============================================================

FINAL_CSV = os.path.join(

    OUTPUT_FOLDER,

    "final_fused_dataset.csv"

)


final_fused_df.to_csv(

    FINAL_CSV,

    index=False,

    encoding="utf-8-sig"

)


print(
    "\nFinal CSV created:"
)


print(
    os.path.abspath(FINAL_CSV)
)


# ============================================================
# STEP 20: SAVE FINAL EXCEL
# ============================================================

FINAL_EXCEL = os.path.join(

    OUTPUT_FOLDER,

    "final_fused_dataset.xlsx"

)


try:

    final_fused_df.to_excel(

        FINAL_EXCEL,

        index=False

    )


    print(
        "\nFinal Excel created:"
    )


    print(
        os.path.abspath(FINAL_EXCEL)
    )


except Exception as e:

    print(
        "\nExcel creation error:",
        e
    )


# ============================================================
# STEP 21: SAVE MATCHING REPORT
# ============================================================

validation_data = {

    "Parameter": [

        "Original records",

        "API records",

        "Combined records",

        "Matching unique keys",

        "Matching original records",

        "Matching API records",

        "Unmatched original records",

        "Unmatched API records",

        "Duplicate keys - Original",

        "Duplicate keys - API",

        "Exact duplicate fused records"

    ],

    "Value": [

        len(original_df),

        len(api_df),

        len(final_fused_df),

        len(matching_keys),

        len(matching_original_records),

        len(matching_api_records),

        len(original_unmatched),

        len(api_unmatched),

        len(original_duplicate_keys),

        len(api_duplicate_keys),

        duplicate_count

    ]

}


validation_df = pd.DataFrame(
    validation_data
)


VALIDATION_CSV = os.path.join(

    OUTPUT_FOLDER,

    "fusion_validation.csv"

)


validation_df.to_csv(

    VALIDATION_CSV,

    index=False

)


print(
    "\nValidation report created:"
)


print(
    os.path.abspath(VALIDATION_CSV)
)


# ============================================================
# STEP 22: SAVE MATCHED RECORDS
# ============================================================

if len(matched_comparison) > 0:

    MATCHED_CSV = os.path.join(

        OUTPUT_FOLDER,

        "matched_records.csv"

    )


    matched_comparison.to_csv(

        MATCHED_CSV,

        index=False

    )


    print(
        "\nMatched records saved:"
    )


    print(
        os.path.abspath(MATCHED_CSV)
    )


# ============================================================
# STEP 23: SAVE UNMATCHED ORIGINAL RECORDS
# ============================================================

ORIGINAL_UNMATCHED_CSV = os.path.join(

    OUTPUT_FOLDER,

    "unmatched_original_records.csv"

)


original_unmatched.to_csv(

    ORIGINAL_UNMATCHED_CSV,

    index=False

)


# ============================================================
# STEP 24: SAVE UNMATCHED API RECORDS
# ============================================================

API_UNMATCHED_CSV = os.path.join(

    OUTPUT_FOLDER,

    "unmatched_api_records.csv"

)


api_unmatched.to_csv(

    API_UNMATCHED_CSV,

    index=False

)


# ============================================================
# STEP 25: FINAL REPORT
# ============================================================

print("\n" + "=" * 70)

print(
    "FINAL FUSION REPORT"
)

print("=" * 70)


print(
    "\nOriginal Dataset:",
    len(original_df),
    "records"
)


print(
    "API Dataset:",
    len(api_df),
    "records"
)


print(
    "Final Fused Dataset:",
    len(final_fused_df),
    "records"
)


print(
    "\nOriginal Attributes:",
    len(original_df.columns)
)


print(
    "API Attributes:",
    len(api_df.columns)
)


print(
    "Final Fused Attributes:",
    len(final_fused_df.columns)
)


print(
    "\nMatching Unique Keys:",
    len(matching_keys)
)


print(
    "Unmatched Original Records:",
    len(original_unmatched)
)


print(
    "Unmatched API Records:",
    len(api_unmatched)
)


print(
    "\nDuplicate Keys - Original:",
    len(original_duplicate_keys)
)


print(
    "Duplicate Keys - API:",
    len(api_duplicate_keys)
)


print(
    "Exact Duplicate Fused Records:",
    duplicate_count
)


print("\nFinal columns:")

print(
    final_fused_df.columns.tolist()
)


print("\nFirst 5 records:")

print(
    final_fused_df.head()
)


print("\n" + "=" * 70)

print(
    "FINAL DATA FUSION COMPLETED SUCCESSFULLY!"
)

print("=" * 70)