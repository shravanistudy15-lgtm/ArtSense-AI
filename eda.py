import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ============================================================
# ARTSENSE AI - PART E
# EXPLORATORY DATA ANALYSIS
# ============================================================

print("=" * 70)
print("ARTSENSE AI - EXPLORATORY DATA ANALYSIS")
print("FINAL FUSED DATASET")
print("=" * 70)


# ============================================================
# STEP 1: PATHS
# ============================================================

DATA_PATH = "fused_data/final_fused_dataset.csv"

OUTPUT_FOLDER = "fused_data/eda_outputs"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# STEP 2: LOAD DATASET
# ============================================================

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )


df = pd.read_csv(DATA_PATH)


print("\nDataset loaded successfully!")

print(
    "Shape:",
    df.shape
)


# ============================================================
# STEP 3: CLEAN SPECIAL MISSING VALUES
# ============================================================

print("\nCleaning missing/unknown values...")


missing_strings = [

    "",

    "nan",

    "NaN",

    "None",

    "[nan]",

    "[NaN]",

    "unknown",

    "Unknown"

]


for column in df.columns:

    df[column] = (

        df[column]

        .astype(str)

        .str.strip()

    )


    df[column] = df[column].replace(

        missing_strings,

        np.nan

    )


# ============================================================
# STEP 4: BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)

print("BASIC DATASET INFORMATION")

print("=" * 70)


print(
    "\nRows:",
    df.shape[0]
)


print(
    "Columns:",
    df.shape[1]
)


print(
    "\nColumns:"
)


print(
    df.columns.tolist()
)


# ============================================================
# STEP 5: SOURCE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)

print("1. DATA SOURCE DISTRIBUTION")

print("=" * 70)


source_counts = (

    df["source"]

    .fillna("Unknown")

    .value_counts()

)


print(
    source_counts
)


plt.figure(
    figsize=(10, 6)
)


source_counts.plot(
    kind="bar"
)


plt.title(
    "Number of Artworks by Data Source"
)


plt.xlabel(
    "Data Source"
)


plt.ylabel(
    "Number of Artworks"
)


plt.xticks(
    rotation=45,
    ha="right"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "01_source_distribution.png"
    ),
    dpi=300
)


plt.close()


# ============================================================
# STEP 6: ORIGINAL VS API DATA
# ============================================================

print("\n" + "=" * 70)

print("2. ORIGINAL VS API RECORDS")

print("=" * 70)


record_source_counts = (

    df["record_source"]

    .fillna("Unknown")

    .value_counts()

)


print(
    record_source_counts
)


plt.figure(
    figsize=(8, 6)
)


record_source_counts.plot(
    kind="bar"
)


plt.title(
    "Original Dataset vs API Dataset"
)


plt.xlabel(
    "Record Source"
)


plt.ylabel(
    "Number of Records"
)


plt.xticks(
    rotation=0
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "02_original_vs_api.png"
    ),
    dpi=300
)


plt.close()


# ============================================================
# STEP 7: ART MOVEMENT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)

print("3. ART MOVEMENT DISTRIBUTION")

print("=" * 70)


if "movement" in df.columns:

    movement_counts = (

        df["movement"]

        .fillna("Unknown")

        .value_counts()

        .head(15)

    )


    print(
        movement_counts
    )


    plt.figure(
        figsize=(11, 7)
    )


    movement_counts.sort_values().plot(
        kind="barh"
    )


    plt.title(
        "Top 15 Art Movements"
    )


    plt.xlabel(
        "Number of Artworks"
    )


    plt.ylabel(
        "Art Movement"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "03_art_movement_distribution.png"
        ),
        dpi=300
    )


    plt.close()


# ============================================================
# STEP 8: ART PERIOD DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)

print("4. ART PERIOD DISTRIBUTION")

print("=" * 70)


if "period" in df.columns:

    period_counts = (

        df["period"]

        .fillna("Unknown")

        .value_counts()

        .head(15)

    )


    print(
        period_counts
    )


    plt.figure(
        figsize=(11, 7)
    )


    period_counts.sort_values().plot(
        kind="barh"
    )


    plt.title(
        "Artworks by Art Period"
    )


    plt.xlabel(
        "Number of Artworks"
    )


    plt.ylabel(
        "Art Period"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "04_art_period_distribution.png"
        ),
        dpi=300
    )


    plt.close()


# ============================================================
# STEP 9: TOP ARTISTS
# ============================================================

print("\n" + "=" * 70)

print("5. TOP ARTISTS")

print("=" * 70)


artist_counts = (

    df["artist"]

    .fillna("Unknown")

    .value_counts()

)


# Remove Unknown
# Remove Unknown
artist_counts = artist_counts[
    artist_counts.index.astype(str).str.lower() != "unknown"
]


top_artists = artist_counts.head(15)


print(
    top_artists
)


plt.figure(
    figsize=(11, 7)
)


top_artists.sort_values().plot(
    kind="barh"
)


plt.title(
    "Top 15 Artists by Number of Artworks"
)


plt.xlabel(
    "Number of Artworks"
)


plt.ylabel(
    "Artist"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "05_top_artists.png"
    ),
    dpi=300
)


plt.close()


# ============================================================
# STEP 10: MEDIUM DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)

print("6. MEDIUM DISTRIBUTION")

print("=" * 70)


if "medium" in df.columns:

    medium_counts = (

        df["medium"]

        .fillna("Unknown")

        .value_counts()

        .head(15)

    )


    print(
        medium_counts
    )


    plt.figure(
        figsize=(11, 7)
    )


    medium_counts.sort_values().plot(
        kind="barh"
    )


    plt.title(
        "Top 15 Artwork Mediums"
    )


    plt.xlabel(
        "Number of Artworks"
    )


    plt.ylabel(
        "Medium"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "06_medium_distribution.png"
        ),
        dpi=300
    )


    plt.close()


# ============================================================
# STEP 11: PRICE CONVERSION
# ============================================================

print("\n" + "=" * 70)

print("7. PRICE ANALYSIS")

print("=" * 70)


if "price" in df.columns:

    price_clean = (

        df["price"]

        .fillna("")

        .astype(str)

        .str.replace(
            "USD",
            "",
            regex=False
        )

        .str.replace(
            ",",
            "",
            regex=False
        )

        .str.strip()

    )


    df["price_numeric"] = pd.to_numeric(

        price_clean,

        errors="coerce"

    )


    valid_prices = (

        df["price_numeric"]

        .dropna()

    )


    print(
        "\nValid prices:",
        len(valid_prices)
    )


    print(
        "\nPrice statistics:"
    )


    print(
        valid_prices.describe()
    )


# ============================================================
# STEP 12: PRICE DISTRIBUTION
# ============================================================

if "price_numeric" in df.columns:

    valid_prices = (

        df["price_numeric"]

        .dropna()

    )


    if len(valid_prices) > 0:

        plt.figure(
            figsize=(10, 6)
        )


        plt.hist(
            valid_prices,
            bins=30
        )


        plt.title(
            "Artwork Price Distribution"
        )


        plt.xlabel(
            "Price (USD)"
        )


        plt.ylabel(
            "Number of Artworks"
        )


        plt.tight_layout()


        plt.savefig(
            os.path.join(
                OUTPUT_FOLDER,
                "07_price_distribution.png"
            ),
            dpi=300
        )


        plt.close()


# ============================================================
# STEP 13: PRICE BOXPLOT
# ============================================================

if "price_numeric" in df.columns:

    valid_prices = (

        df["price_numeric"]

        .dropna()

    )


    if len(valid_prices) > 0:

        plt.figure(
            figsize=(10, 5)
        )


        plt.boxplot(
            valid_prices
        )


        plt.title(
            "Artwork Price Boxplot"
        )


        plt.ylabel(
            "Price (USD)"
        )


        plt.tight_layout()


        plt.savefig(
            os.path.join(
                OUTPUT_FOLDER,
                "08_price_boxplot.png"
            ),
            dpi=300
        )


        plt.close()


# ============================================================
# STEP 14: TOP 10 MOST EXPENSIVE ARTWORKS
# ============================================================

print("\n" + "=" * 70)

print("8. TOP 10 MOST EXPENSIVE ARTWORKS")

print("=" * 70)


if "price_numeric" in df.columns:

    expensive = (

        df[

            [

                "title",

                "artist",

                "price_numeric",

                "source"

            ]

        ]

        .dropna(
            subset=[
                "price_numeric"
            ]
        )

        .sort_values(
            "price_numeric",
            ascending=False
        )

        .head(10)

    )


    print(
        expensive.to_string(
            index=False
        )
    )


    plt.figure(
        figsize=(12, 7)
    )


    labels = (

        expensive["title"]

        .astype(str)

        .str[:35]

    )


    plt.barh(
        labels,
        expensive["price_numeric"]
    )


    plt.title(
        "Top 10 Most Expensive Artworks"
    )


    plt.xlabel(
        "Price (USD)"
    )


    plt.ylabel(
        "Artwork"
    )


    plt.gca().invert_yaxis()


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "09_top_10_expensive.png"
        ),
        dpi=300
    )


    plt.close()


# ============================================================
# STEP 15: API SOURCE VS MEDIUM
# ============================================================

print("\n" + "=" * 70)

print("9. API SOURCE VS MEDIUM")

print("=" * 70)


if (

    "source" in df.columns

    and

    "medium" in df.columns

):

    cross_tab = pd.crosstab(

        df["source"].fillna("Unknown"),

        df["medium"].fillna("Unknown")

    )


    # Keep top 10 mediums
    top_mediums = (

        df["medium"]

        .fillna("Unknown")

        .value_counts()

        .head(10)

        .index

    )


    cross_tab = cross_tab[
        [
            col
            for col in cross_tab.columns
            if col in top_mediums
        ]

    ]


    print(
        cross_tab
    )


    plt.figure(
        figsize=(13, 7)
    )


    cross_tab.plot(
        kind="bar",
        figsize=(13, 7)
    )


    plt.title(
        "Data Source vs Artwork Medium"
    )


    plt.xlabel(
        "Data Source"
    )


    plt.ylabel(
        "Number of Artworks"
    )


    plt.xticks(
        rotation=45,
        ha="right"
    )


    plt.legend(
        title="Medium",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "10_source_vs_medium.png"
        ),
        dpi=300
    )


    plt.close()


# ============================================================
# STEP 16: API SOURCE VS ART PERIOD
# ============================================================

print("\n" + "=" * 70)

print("10. DATA SOURCE VS ART PERIOD")

print("=" * 70)


if (

    "source" in df.columns

    and

    "period" in df.columns

):

    source_period = pd.crosstab(

        df["source"].fillna("Unknown"),

        df["period"].fillna("Unknown")

    )


    top_periods = (

        df["period"]

        .fillna("Unknown")

        .value_counts()

        .head(10)

        .index

    )


    source_period = source_period[
        [

            col

            for col in source_period.columns

            if col in top_periods

        ]

    ]


    print(
        source_period
    )


    plt.figure(
        figsize=(13, 7)
    )


    source_period.plot(
        kind="bar",
        figsize=(13, 7)
    )


    plt.title(
        "Data Source vs Art Period"
    )


    plt.xlabel(
        "Data Source"
    )


    plt.ylabel(
        "Number of Artworks"
    )


    plt.xticks(
        rotation=45,
        ha="right"
    )


    plt.legend(
        title="Art Period",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "11_source_vs_period.png"
        ),
        dpi=300
    )


    plt.close()


# ============================================================
# STEP 17: CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("11. CORRELATION ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# Create numerical creation year
# ------------------------------------------------------------

if "date" in df.columns:

    df["creation_year"] = pd.to_numeric(
        df["date"].astype(str).str.extract(
            r"(\d{4})",
            expand=False
        ),
        errors="coerce"
    )

else:

    df["creation_year"] = np.nan


# ------------------------------------------------------------
# Make sure price is numeric
# ------------------------------------------------------------

if "price_numeric" not in df.columns:

    price_clean = (

        df["price"]
        .fillna("")
        .astype(str)
        .str.replace(
            "USD",
            "",
            regex=False
        )
        .str.replace(
            ",",
            "",
            regex=False
        )
        .str.strip()

    )

    df["price_numeric"] = pd.to_numeric(
        price_clean,
        errors="coerce"
    )


# ------------------------------------------------------------
# Select numerical variables
# ------------------------------------------------------------

correlation_data = df[
    [
        "price_numeric",
        "creation_year"
    ]
].copy()


print("\nNumerical variables used:")

print(
    correlation_data.columns.tolist()
)


# ------------------------------------------------------------
# Calculate correlation
# ------------------------------------------------------------

correlation_matrix = (
    correlation_data
    .corr()
)


print("\nCorrelation matrix:")

print(
    correlation_matrix
)


# ------------------------------------------------------------
# Create heatmap
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)


sns.heatmap(

    correlation_matrix,

    annot=True,

    fmt=".2f",

    cmap="viridis",

    linewidths=0.5

)


plt.title(
    "Correlation Heatmap: Price and Creation Year"
)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_FOLDER,

        "12_correlation_heatmap.png"

    ),

    dpi=300

)


plt.close()


print(
    "\nCorrelation heatmap created successfully!"
)

print(
    "Saved as: 12_correlation_heatmap.png"
)


# ============================================================
# STEP 18: API DATA COMPLETENESS
# ============================================================

print("\n" + "=" * 70)

print("12. API DATA COMPLETENESS")

print("=" * 70)


api_only = df[
    df["record_source"]
    == "API Dataset"
]


completeness = (

    api_only.notna()

    .mean()

    * 100

)


completeness = (

    completeness

    .sort_values()

)


print(
    "\nAPI attribute completeness (%):"
)


print(
    completeness
)


plt.figure(
    figsize=(11, 8)
)


completeness.plot(
    kind="barh"
)


plt.title(
    "API Dataset Attribute Completeness"
)


plt.xlabel(
    "Completeness (%)"
)


plt.ylabel(
    "Attribute"
)


plt.xlim(
    0,
    100
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "13_api_completeness.png"
    ),
    dpi=300
)


plt.close()


# ============================================================
# STEP 19: SAVE EDA SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("SAVING EDA SUMMARY")

print("=" * 70)


summary = {

    "Total Records":
        len(df),

    "Total Attributes":
        len(df.columns),

    "Original Records":
        (
            df["record_source"]
            == "Original Dataset"
        ).sum(),

    "API Records":
        (
            df["record_source"]
            == "API Dataset"
        ).sum(),

    "Unique Artists":
        df["artist"]
        .nunique(),

    "Unique Titles":
        df["title"]
        .nunique(),

    "Unique Sources":
        df["source"]
        .nunique(),

    "Unique Movements":
        (
            df["movement"]
            .nunique()
        ),

    "Unique Periods":
        (
            df["period"]
            .nunique()
        )

}


summary_df = pd.DataFrame({

    "Metric":
        list(summary.keys()),

    "Value":
        list(summary.values())

})


SUMMARY_PATH = os.path.join(

    OUTPUT_FOLDER,

    "eda_summary.csv"

)


summary_df.to_csv(

    SUMMARY_PATH,

    index=False

)


print(
    "\nEDA summary saved:"
)


print(
    os.path.abspath(
        SUMMARY_PATH
    )
)


# ============================================================
# STEP 20: FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)

print(
    "EDA COMPLETED SUCCESSFULLY!"
)

print("=" * 70)


print(
    "\nAll graphs saved in:"
)


print(
    os.path.abspath(
        OUTPUT_FOLDER
    )
)


print("\nGenerated files:")


for filename in sorted(
    os.listdir(OUTPUT_FOLDER)
):

    print(
        " -",
        filename
    )


print("\n" + "=" * 70)