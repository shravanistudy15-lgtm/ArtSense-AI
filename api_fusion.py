import pandas as pd
import os


# ==================================================
# ARTSENSE AI - API DATA FUSION
# ==================================================

print("=" * 60)
print("ARTSENSE AI - API DATA FUSION")
print("=" * 60)


# ==================================================
# STEP 1: CREATE OUTPUT FOLDER
# ==================================================

OUTPUT_FOLDER = "fused_data"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ==================================================
# STEP 2: API DATA FILES
# ==================================================

api_files = {

    "Met Museum":
        "api_data/met_artworks.csv",

    "Smithsonian":
        "api_data/smithsonian_artworks.csv",

    "Art Institute of Chicago":
        "api_data/chicago_artworks.csv",

    "Cleveland Museum":
        "api_data/cleveland_artworks.csv",

    "Victoria and Albert Museum":
        "api_data/va_artworks.csv"

}


# ==================================================
# STEP 3: HELPER FUNCTION
# REMOVE / MERGE DUPLICATE COLUMN NAMES
# ==================================================

def remove_duplicate_columns(df):

    # Create a new dataframe
    new_df = pd.DataFrame(index=df.index)

    # Go through every column name
    for column in df.columns:

        # If column name appears for the first time
        if column not in new_df.columns:

            # If there is only one column with this name
            same_columns = df.loc[:, df.columns == column]

            if same_columns.shape[1] == 1:

                new_df[column] = same_columns.iloc[:, 0]

            else:

                # Multiple columns have same name
                # Take first non-empty value
                merged_series = same_columns.iloc[:, 0]

                for i in range(1, same_columns.shape[1]):

                    merged_series = merged_series.combine_first(
                        same_columns.iloc[:, i]
                    )

                new_df[column] = merged_series

    return new_df


# ==================================================
# STEP 4: COLUMN STANDARDIZATION
# ==================================================

def standardize_columns(df, source):

    print("  Original columns:")
    print(" ", df.columns.tolist())

    # Make a copy
    df = df.copy()

    # Make all column names lowercase
    df.columns = [

        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")

        for column in df.columns

    ]


    # ----------------------------------------------
    # POSSIBLE COLUMN NAMES -> STANDARD NAMES
    # ----------------------------------------------

    column_mapping = {

        # TITLE
        "title": "title",
        "object_title": "title",
        "artwork_title": "title",
        "object_name": "title",
        "name": "title",

        # ARTIST
        "artist": "artist",
        "artist_display": "artist",
        "creator": "artist",
        "maker": "artist",
        "artist_name": "artist",
        "artistdisplayname": "artist",
        "artist_name_display": "artist",

        # DATE
        "date": "date",
        "object_date": "date",
        "creation_date": "date",
        "date_created": "date",
        "year": "date",
        "yearcreation": "date",
        "year_creation": "date",

        # MEDIUM
        "medium": "medium",
        "materials": "medium",
        "material": "medium",
        "technique": "medium",

        # DEPARTMENT
        "department": "department",
        "collection": "department",
        "department_title": "department",
        "departmenttitle": "department",

        # CLASSIFICATION
        "classification": "classification",
        "object_type": "classification",
        "type": "classification",
        "category": "classification",
        "object_name_type": "classification",

        # CULTURE
        "culture": "culture",

        # DESCRIPTION
        "description": "description",
        "object_description": "description",
        "summary": "description",

        # IMAGE URL
        "image_url": "image_url",
        "imageurl": "image_url",
        "primaryimage": "image_url",
        "primary_image": "image_url",
        "image": "image_url",
        "image_link": "image_url",
        "iiif_url": "image_url",

        # OBJECT ID
        "object_id": "object_id",
        "objectid": "object_id",
        "api_id": "object_id",
        "id": "object_id",
        "accession_number": "object_id",
        "accessionnumber": "object_id"

    }


    # ----------------------------------------------
    # RENAME COLUMNS
    # ----------------------------------------------

    rename_dict = {}

    for column in df.columns:

        if column in column_mapping:

            rename_dict[column] = column_mapping[column]


    df = df.rename(
        columns=rename_dict
    )


    # ----------------------------------------------
    # IMPORTANT:
    # MERGE DUPLICATE COLUMN NAMES
    # ----------------------------------------------

    df = remove_duplicate_columns(df)


    # ----------------------------------------------
    # STANDARD COLUMNS
    # ----------------------------------------------

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
        "object_id"

    ]


    # ----------------------------------------------
    # ADD MISSING COLUMNS
    # ----------------------------------------------

    for column in standard_columns:

        if column not in df.columns:

            df[column] = "Unknown"


    # ----------------------------------------------
    # ADD SOURCE
    # ----------------------------------------------

    df["source"] = source


    # ----------------------------------------------
    # KEEP ONLY STANDARD COLUMNS
    # ----------------------------------------------

    df = df[
        standard_columns + ["source"]
    ].copy()


    # Final duplicate column safety check
    df = remove_duplicate_columns(df)


    # Reset row index
    df = df.reset_index(
        drop=True
    )


    print("  Standardized columns:")
    print(" ", df.columns.tolist())


    return df


# ==================================================
# STEP 5: LOAD AND STANDARDIZE ALL APIs
# ==================================================

all_datasets = []


for source, file_path in api_files.items():

    print("\n" + "-" * 60)
    print("Loading:", source)
    print("File:", file_path)
    print("-" * 60)


    # Check file exists
    if not os.path.exists(file_path):

        print(
            f"WARNING: File not found: {file_path}"
        )

        continue


    try:

        # Read CSV
        df = pd.read_csv(
            file_path
        )


        print(
            "Original shape:",
            df.shape
        )


        # Standardize dataset
        standardized_df = standardize_columns(
            df,
            source
        )


        print(
            "Standardized shape:",
            standardized_df.shape
        )


        # Add to dataset list
        all_datasets.append(
            standardized_df
        )


        print(
            f"{source} added successfully!"
        )


    except Exception as e:

        print(
            f"Error processing {source}: {e}"
        )


# ==================================================
# STEP 6: CHECK DATASETS
# ==================================================

if not all_datasets:

    raise ValueError(
        "No API datasets were successfully loaded."
    )


print("\n" + "=" * 60)
print("DATASETS READY FOR FUSION")
print("=" * 60)


for i, dataset in enumerate(
    all_datasets,
    start=1
):

    print(
        f"\nDataset {i}:"
    )

    print(
        "Shape:",
        dataset.shape
    )

    print(
        "Columns:",
        dataset.columns.tolist()
    )


# ==================================================
# STEP 7: FINAL SAFETY CLEANING
# ==================================================

print("\nCleaning datasets before fusion...")


clean_datasets = []


for i, dataset in enumerate(
    all_datasets,
    start=1
):

    # Make copy
    dataset = dataset.copy()


    # Remove duplicate columns again
    dataset = remove_duplicate_columns(
        dataset
    )


    # Reset index
    dataset = dataset.reset_index(
        drop=True
    )


    print(
        f"Dataset {i} cleaned successfully."
    )


    clean_datasets.append(
        dataset
    )


# Replace original list
all_datasets = clean_datasets


# ==================================================
# STEP 8: COMBINE ALL DATASETS
# ==================================================

print("\n" + "=" * 60)
print("COMBINING ALL API DATASETS")
print("=" * 60)


fused_df = pd.concat(

    all_datasets,

    ignore_index=True,

    sort=False

)


print(
    "\nCombined shape:",
    fused_df.shape
)


# ==================================================
# STEP 9: CLEAN FUSED DATASET
# ==================================================

print("\nCleaning fused dataset...")


# Remove completely empty rows
fused_df = fused_df.dropna(
    how="all"
)


# Replace missing values
fused_df = fused_df.fillna(
    "Unknown"
)


# Important text columns
text_columns = [

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
    "source"

]


# Convert columns to clean strings
for column in text_columns:

    if column in fused_df.columns:

        fused_df[column] = (

            fused_df[column]

            .astype(str)

            .str.strip()

        )


# Replace empty strings with Unknown
for column in text_columns:

    if column in fused_df.columns:

        fused_df[column] = fused_df[column].replace(

            "",

            "Unknown"

        )


# ==================================================
# STEP 10: REMOVE DUPLICATE RECORDS
# ==================================================

print("\nRemoving duplicate records...")


before_duplicates = len(
    fused_df
)


fused_df = fused_df.drop_duplicates(

    subset=[

        "title",

        "artist",

        "source"

    ],

    keep="first"

)


after_duplicates = len(
    fused_df
)


print(
    "Duplicates removed:",

    before_duplicates - after_duplicates
)


# ==================================================
# STEP 11: REMOVE EMPTY / UNKNOWN TITLES
# ==================================================

print("\nRemoving records without valid titles...")


before_title_cleaning = len(
    fused_df
)


fused_df = fused_df[

    fused_df["title"]

    .notna()

]


fused_df = fused_df[

    fused_df["title"]

    .astype(str)

    .str.strip()

    .str.lower()

    .ne("unknown")

]


fused_df = fused_df[

    fused_df["title"]

    .astype(str)

    .str.strip()

    .ne("")

]


after_title_cleaning = len(
    fused_df
)


print(
    "Records removed due to invalid title:",

    before_title_cleaning - after_title_cleaning
)


# ==================================================
# STEP 12: RESET INDEX
# ==================================================

fused_df = fused_df.reset_index(

    drop=True

)


# ==================================================
# STEP 13: SAVE CSV
# ==================================================

csv_output = os.path.join(

    OUTPUT_FOLDER,

    "artsense_fused_dataset.csv"

)


fused_df.to_csv(

    csv_output,

    index=False,

    encoding="utf-8-sig"

)


print("\nCSV file created successfully!")

print(
    "Location:",
    os.path.abspath(csv_output)
)


# ==================================================
# STEP 14: SAVE EXCEL
# ==================================================

excel_output = os.path.join(

    OUTPUT_FOLDER,

    "artsense_fused_dataset.xlsx"

)


try:

    fused_df.to_excel(

        excel_output,

        index=False

    )


    print(
        "\nExcel file created successfully!"
    )


    print(
        "Location:",
        os.path.abspath(excel_output)
    )


except Exception as e:

    print(
        "\nCould not create Excel file."
    )


    print(
        "Error:",
        e
    )


# ==================================================
# STEP 15: SAVE SOURCE-WISE SUMMARY
# ==================================================

summary_df = (

    fused_df["source"]

    .value_counts()

    .reset_index()

)


summary_df.columns = [

    "Source",

    "Number of Artworks"

]


summary_output = os.path.join(

    OUTPUT_FOLDER,

    "api_fusion_summary.csv"

)


summary_df.to_csv(

    summary_output,

    index=False

)


print(
    "\nFusion summary created!"
)


print(
    "Location:",
    os.path.abspath(summary_output)
)


# ==================================================
# STEP 16: SHOW FINAL DATASET INFORMATION
# ==================================================

print("\n" + "=" * 60)

print(
    "FINAL FUSED DATASET INFORMATION"
)

print("=" * 60)


print(
    "\nFinal Shape:",

    fused_df.shape
)


print(
    "\nFinal Columns:"
)


print(
    fused_df.columns.tolist()
)


print(
    "\nRecords by Source:"
)


print(
    fused_df["source"]

    .value_counts()
)


print(
    "\nMissing Values:"
)


print(
    fused_df.isnull().sum()
)


print(
    "\nFirst 10 Rows:"
)


print(
    fused_df.head(10)
)


print("\n" + "=" * 60)

print(
    "API FUSION COMPLETED SUCCESSFULLY!"
)

print("=" * 60)