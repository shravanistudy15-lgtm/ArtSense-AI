import pandas as pd
import numpy as np
import os


# ============================================================
# ARTSENSE AI - PART D
# DATA QUALITY ASSESSMENT
# ============================================================

print("=" * 70)
print("ARTSENSE AI - DATA QUALITY ASSESSMENT")
print("FINAL FUSED DATASET")
print("=" * 70)


# ============================================================
# STEP 1: FILE PATH
# ============================================================

DATA_PATH = "fused_data/final_fused_dataset.csv"

OUTPUT_FOLDER = "fused_data"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# STEP 2: CHECK FILE
# ============================================================

print("\nChecking dataset...")

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

print("Dataset found:")
print(os.path.abspath(DATA_PATH))


# ============================================================
# STEP 3: LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("1. DATASET DIMENSIONS")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nRows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nDataset shape:", df.shape)


# ============================================================
# STEP 4: COLUMN INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("2. COLUMN INFORMATION")
print("=" * 70)

print("\nColumns:")

for i, column in enumerate(df.columns, start=1):

    print(
        f"{i}. {column}"
    )


# ============================================================
# STEP 5: DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("3. DATA TYPES")
print("=" * 70)

dtype_df = pd.DataFrame({

    "column": df.columns,

    "data_type": [
        str(dtype)
        for dtype in df.dtypes
    ]

})

print(dtype_df.to_string(index=False))


# ============================================================
# STEP 6: MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("4. MISSING VALUES")
print("=" * 70)

missing_count = df.isnull().sum()

missing_percentage = (
    df.isnull().mean() * 100
)


missing_df = pd.DataFrame({

    "column": df.columns,

    "missing_count": missing_count.values,

    "missing_percentage": (
        missing_percentage.values
    )

})


print(
    missing_df.to_string(
        index=False
    )
)


# ============================================================
# STEP 7: FIND COLUMN WITH MAXIMUM MISSING VALUES
# ============================================================

max_missing_column = (
    missing_df
    .sort_values(
        "missing_count",
        ascending=False
    )
    .iloc[0]
)


print(
    "\nColumn with maximum missing values:"
)

print(
    max_missing_column["column"]
)

print(
    "Missing values:",
    int(max_missing_column["missing_count"])
)

print(
    "Missing percentage:",
    round(
        max_missing_column[
            "missing_percentage"
        ],
        2
    ),
    "%"
)


# ============================================================
# STEP 8: DUPLICATE RECORDS
# ============================================================

print("\n" + "=" * 70)
print("5. DUPLICATE RECORD ANALYSIS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

duplicate_percentage = (
    duplicate_count
    / len(df)
    * 100
)


print(
    "\nDuplicate records:",
    duplicate_count
)

print(
    "Duplicate percentage:",
    round(
        duplicate_percentage,
        2
    ),
    "%"
)


# ============================================================
# STEP 9: UNIQUE VALUES
# ============================================================

print("\n" + "=" * 70)
print("6. UNIQUE VALUE ANALYSIS")
print("=" * 70)


unique_df = pd.DataFrame({

    "column": df.columns,

    "unique_values": [
        df[column].nunique(
            dropna=False
        )
        for column in df.columns
    ]

})


print(
    unique_df.to_string(
        index=False
    )
)


# ============================================================
# STEP 10: CATEGORICAL CONSISTENCY
# ============================================================

print("\n" + "=" * 70)
print("7. CATEGORICAL CONSISTENCY")
print("=" * 70)


categorical_columns = [

    "source",

    "record_source",

    "period",

    "movement",

    "medium",

    "department",

    "classification",

    "culture"

]


for column in categorical_columns:

    if column in df.columns:

        print(
            f"\n--- {column} ---"
        )

        values = (

            df[column]

            .astype(str)

            .str.strip()

            .value_counts()

            .head(15)

        )

        print(values)


# ============================================================
# STEP 11: TEXT CONSISTENCY CHECK
# ============================================================

print("\n" + "=" * 70)
print("8. TEXT CONSISTENCY CHECK")
print("=" * 70)


text_columns = [

    "title",

    "artist",

    "medium",

    "department",

    "classification",

    "culture",

    "source"

]


text_quality_results = []


for column in text_columns:

    if column not in df.columns:

        continue


    series = (
        df[column]
        .astype(str)
        .str.strip()
    )


    empty_count = (
        series
        .isin([
            "",
            "Unknown",
            "nan",
            "None"
        ])
        .sum()
    )


    leading_trailing_spaces = (
        df[column]
        .astype(str)
        .apply(
            lambda x:
            x != x.strip()
        )
        .sum()
    )


    text_quality_results.append({

        "column": column,

        "empty_or_unknown": (
            empty_count
        ),

        "leading_trailing_spaces": (
            leading_trailing_spaces
        )

    })


text_quality_df = pd.DataFrame(
    text_quality_results
)


print(
    text_quality_df.to_string(
        index=False
    )
)


# ============================================================
# STEP 12: DATE / YEAR QUALITY
# ============================================================

print("\n" + "=" * 70)
print("9. DATE / YEAR CONSISTENCY")
print("=" * 70)


if "date" in df.columns:

    print(
        "\nSample date values:"
    )

    print(
        df["date"]
        .value_counts()
        .head(20)
    )


if "yearCreation" in df.columns:

    print(
        "\nOriginal yearCreation values:"
    )

    print(
        df["yearCreation"]
        .value_counts()
        .head(20)
    )


# ============================================================
# STEP 13: NUMERICAL DATA ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("10. NUMERICAL DATA ANALYSIS")
print("=" * 70)


numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


if numeric_columns:

    print(
        "\nNumerical columns:"
    )

    print(
        numeric_columns
    )


    numeric_summary = (
        df[numeric_columns]
        .describe()
        .T
    )


    print(
        "\nStatistical summary:"
    )

    print(
        numeric_summary
    )

else:

    print(
        "\nNo directly numeric columns found."
    )


# ============================================================
# STEP 14: PRICE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("11. PRICE QUALITY CHECK")
print("=" * 70)


if "price" in df.columns:

    price_clean = (

        df["price"]

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


    price_numeric = pd.to_numeric(
        price_clean,
        errors="coerce"
    )


    print(
        "\nValid numeric prices:",
        price_numeric.notna().sum()
    )


    print(
        "Invalid/missing prices:",
        price_numeric.isna().sum()
    )


    if price_numeric.notna().sum() > 0:

        print(
            "\nPrice statistics:"
        )

        print(
            price_numeric.describe()
        )


# ============================================================
# STEP 15: OUTLIER DETECTION USING IQR
# ============================================================

print("\n" + "=" * 70)
print("12. OUTLIER DETECTION")
print("METHOD: IQR")
print("=" * 70)


outlier_results = []


if "price" in df.columns:

    price_clean = (

        df["price"]

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


    price_numeric = pd.to_numeric(
        price_clean,
        errors="coerce"
    )


    valid_prices = (
        price_numeric
        .dropna()
    )


    if len(valid_prices) > 0:

        Q1 = valid_prices.quantile(
            0.25
        )

        Q3 = valid_prices.quantile(
            0.75
        )


        IQR = Q3 - Q1


        lower_bound = (
            Q1 - 1.5 * IQR
        )

        upper_bound = (
            Q3 + 1.5 * IQR
        )


        outliers = valid_prices[
            (valid_prices < lower_bound)
            |
            (valid_prices > upper_bound)
        ]


        print(
            "\nPrice Q1:",
            Q1
        )

        print(
            "Price Q3:",
            Q3
        )

        print(
            "IQR:",
            IQR
        )

        print(
            "Lower bound:",
            lower_bound
        )

        print(
            "Upper bound:",
            upper_bound
        )

        print(
            "Number of price outliers:",
            len(outliers)
        )


        outlier_results.append({

            "column": "price",

            "Q1": Q1,

            "Q3": Q3,

            "IQR": IQR,

            "lower_bound": lower_bound,

            "upper_bound": upper_bound,

            "outlier_count": len(outliers)

        })


# ============================================================
# STEP 16: SOURCE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("13. RECORDS BY SOURCE")
print("=" * 70)


if "source" in df.columns:

    source_counts = (
        df["source"]
        .value_counts()
    )


    print(
        source_counts
    )


# ============================================================
# STEP 17: RECORD SOURCE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("14. ORIGINAL VS API RECORDS")
print("=" * 70)


if "record_source" in df.columns:

    record_source_counts = (
        df["record_source"]
        .value_counts()
    )


    print(
        record_source_counts
    )


# ============================================================
# STEP 18: SAVE QUALITY REPORT
# ============================================================

print("\n" + "=" * 70)
print("15. SAVING REPORTS")
print("=" * 70)


quality_report = pd.DataFrame({

    "Metric": [

        "Total records",

        "Total attributes",

        "Duplicate records",

        "Duplicate percentage",

        "Total missing values",

        "Columns with missing values",

        "Original records",

        "API records"

    ],

    "Value": [

        len(df),

        len(df.columns),

        duplicate_count,

        round(
            duplicate_percentage,
            2
        ),

        int(
            df.isnull()
            .sum()
            .sum()
        ),

        int(
            (
                df.isnull()
                .sum() > 0
            ).sum()
        ),

        (
            (
                df["record_source"]
                == "Original Dataset"
            ).sum()
            if "record_source" in df.columns
            else 0
        ),

        (
            (
                df["record_source"]
                == "API Dataset"
            ).sum()
            if "record_source" in df.columns
            else 0
        )

    ]

})


QUALITY_REPORT_PATH = os.path.join(

    OUTPUT_FOLDER,

    "data_quality_summary.csv"

)


quality_report.to_csv(

    QUALITY_REPORT_PATH,

    index=False

)


print(
    "\nQuality summary saved:"
)

print(
    os.path.abspath(
        QUALITY_REPORT_PATH
    )
)


# ============================================================
# STEP 19: SAVE MISSING VALUE REPORT
# ============================================================

MISSING_REPORT_PATH = os.path.join(

    OUTPUT_FOLDER,

    "missing_value_report.csv"

)


missing_df.to_csv(

    MISSING_REPORT_PATH,

    index=False

)


print(
    "\nMissing value report saved:"
)

print(
    os.path.abspath(
        MISSING_REPORT_PATH
    )
)


# ============================================================
# STEP 20: SAVE UNIQUE VALUE REPORT
# ============================================================

UNIQUE_REPORT_PATH = os.path.join(

    OUTPUT_FOLDER,

    "unique_value_report.csv"

)


unique_df.to_csv(

    UNIQUE_REPORT_PATH,

    index=False

)


print(
    "\nUnique value report saved:"
)

print(
    os.path.abspath(
        UNIQUE_REPORT_PATH
    )
)


# ============================================================
# STEP 21: SAVE OUTLIER REPORT
# ============================================================

if outlier_results:

    outlier_df = pd.DataFrame(
        outlier_results
    )


    OUTLIER_REPORT_PATH = os.path.join(

        OUTPUT_FOLDER,

        "outlier_report.csv"

    )


    outlier_df.to_csv(

        OUTLIER_REPORT_PATH,

        index=False

    )


    print(
        "\nOutlier report saved:"
    )

    print(
        os.path.abspath(
            OUTLIER_REPORT_PATH
        )
    )


# ============================================================
# STEP 22: FINAL DATA QUALITY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATA QUALITY SUMMARY")
print("=" * 70)


print(
    "\nTotal records:",
    len(df)
)


print(
    "Total attributes:",
    len(df.columns)
)


print(
    "Duplicate records:",
    duplicate_count
)


print(
    "Total missing values:",
    int(
        df.isnull()
        .sum()
        .sum()
    )
)


print(
    "Columns with missing values:",
    int(
        (
            df.isnull()
            .sum() > 0
        ).sum()
    )
)


if "record_source" in df.columns:

    print(
        "Original records:",
        (
            df["record_source"]
            == "Original Dataset"
        ).sum()
    )


    print(
        "API records:",
        (
            df["record_source"]
            == "API Dataset"
        ).sum()
    )


print("\n" + "=" * 70)

print(
    "DATA QUALITY ASSESSMENT COMPLETED!"
)

print("=" * 70)