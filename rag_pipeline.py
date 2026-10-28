import pandas as pd
import os
import shutil
import hashlib

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# ARTSENSE AI - FINAL LLM + RAG PIPELINE
# ============================================================

print("=" * 70)
print("ARTSENSE AI - FINAL RAG PIPELINE")
print("957-RECORD FUSED DATASET + API DATA")
print("=" * 70)


# ============================================================
# STEP 1: PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "fused_data",
    "final_fused_dataset.csv"
)

VECTOR_DB_PATH = os.path.join(
    BASE_DIR,
    "vector_db"
)

SIGNATURE_FILE = os.path.join(
    VECTOR_DB_PATH,
    "dataset_signature.txt"
)


print("\nProject directory:")
print(BASE_DIR)

print("\nUsing FINAL dataset:")
print(DATA_PATH)


# ============================================================
# STEP 2: CHECK DATASET
# ============================================================

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        "\nFinal fused dataset not found!\n\n"
        "Expected file:\n"
        f"{DATA_PATH}\n\n"
        "Please run:\n"
        "python final_fusion.py"
    )


# ============================================================
# STEP 3: LOAD FINAL DATASET
# ============================================================

df = pd.read_csv(
    DATA_PATH
)

print("\nDataset loaded successfully!")

print(
    "Original shape:",
    df.shape
)


# ============================================================
# STEP 4: REMOVE UNNECESSARY INDEX COLUMN
# ============================================================

if "Unnamed: 0" in df.columns:

    df = df.drop(
        columns=["Unnamed: 0"]
    )


# ============================================================
# STEP 5: CHECK REQUIRED COLUMNS
# ============================================================

print("\nChecking dataset columns...")

required_columns = [
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

for column in required_columns:

    if column not in df.columns:

        df[column] = "Unknown"


print("All required columns available.")


# ============================================================
# STEP 6: CLEAN DATA
# ============================================================

print("\nCleaning dataset...")


def clean_value(value):

    if pd.isna(value):

        return "Unknown"

    value = str(value).strip()

    if not value:

        return "Unknown"

    if value.lower() in [
        "nan",
        "none",
        "null"
    ]:

        return "Unknown"

    return value


for column in required_columns:

    df[column] = (
        df[column]
        .apply(clean_value)
    )


print(
    "Final shape:",
    df.shape
)


# ============================================================
# STEP 7: SHOW DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET INFORMATION")
print("=" * 70)

print(
    "\nRows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)

print(
    "\nColumns:"
)

print(
    df.columns.tolist()
)


print("\nRecords by source:")

print(
    df["source"].value_counts()
)


print("\nRecords by record source:")

print(
    df["record_source"].value_counts()
)


# ============================================================
# STEP 8: CREATE ARTWORK TEXT
# ============================================================

def create_artwork_text(row):

    text = f"""
Artwork Title: {row['title']}

Artist: {row['artist']}

Date / Year of Creation: {row['date']}

Medium / Materials: {row['medium']}

Department: {row['department']}

Classification: {row['classification']}

Culture: {row['culture']}

Description: {row['description']}

Data Source: {row['source']}

Record Source: {row['record_source']}

Object ID: {row['object_id']}

Price: {row['price']}

Signed: {row['signed']}

Condition: {row['condition']}

Art Period: {row['period']}

Art Movement: {row['movement']}

Image URL: {row['image_url']}
"""

    return text.strip()


print("\nCreating artwork documents...")


df["artwork_text"] = df.apply(
    create_artwork_text,
    axis=1
)


# ============================================================
# STEP 9: CREATE LANGCHAIN DOCUMENTS
# ============================================================

documents = []


for index, row in df.iterrows():

    doc = Document(

        page_content=row[
            "artwork_text"
        ],

        metadata={

            "id": int(index),

            "title": row[
                "title"
            ],

            "artist": row[
                "artist"
            ],

            "source": row[
                "source"
            ],

            "record_source": row[
                "record_source"
            ],

            "movement": row[
                "movement"
            ],

            "period": row[
                "period"
            ],

            "medium": row[
                "medium"
            ],

            "date": row[
                "date"
            ],

            "price": row[
                "price"
            ]
        }
    )

    documents.append(
        doc
    )


print(
    "\nDocuments created successfully!"
)

print(
    "Total artwork documents:",
    len(documents)
)


# ============================================================
# STEP 10: CREATE DATASET SIGNATURE
# ============================================================

print(
    "\nCreating dataset signature..."
)


def get_dataset_signature():

    with open(
        DATA_PATH,
        "rb"
    ) as file:

        file_bytes = file.read()

    return hashlib.md5(
        file_bytes
    ).hexdigest()


dataset_signature = (
    get_dataset_signature()
)


print(
    "Dataset signature:",
    dataset_signature
)


# ============================================================
# STEP 11: LOAD HUGGINGFACE EMBEDDING MODEL
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "LOADING HUGGINGFACE EMBEDDING MODEL"
)

print(
    "=" * 70
)


embeddings = HuggingFaceEmbeddings(

    model_name=
        "sentence-transformers/all-MiniLM-L6-v2"
)


print(
    "Embedding model loaded successfully!"
)


# ============================================================
# STEP 12: CHECK EXISTING FAISS DATABASE
# ============================================================

faiss_index_path = os.path.join(
    VECTOR_DB_PATH,
    "index.faiss"
)

faiss_pickle_path = os.path.join(
    VECTOR_DB_PATH,
    "index.pkl"
)


existing_database = (

    os.path.exists(
        faiss_index_path
    )

    and

    os.path.exists(
        faiss_pickle_path
    )
)


old_dataset = False


if existing_database:

    if os.path.exists(
        SIGNATURE_FILE
    ):

        with open(
            SIGNATURE_FILE,
            "r"
        ) as file:

            old_signature = (
                file.read().strip()
            )

        if old_signature != dataset_signature:

            old_dataset = True

            print(
                "\nDataset has changed."
            )

            print(
                "Old FAISS database will be rebuilt."
            )

    else:

        old_dataset = True

        print(
            "\nExisting FAISS database has no dataset signature."
        )

        print(
            "Assuming it belongs to the old dataset."
        )

else:

    print(
        "\nNo existing FAISS database found."
    )


# ============================================================
# STEP 13: REBUILD FAISS IF NECESSARY
# ============================================================

if old_dataset:

    print(
        "\nRemoving old FAISS database..."
    )

    try:

        shutil.rmtree(
            VECTOR_DB_PATH
        )

        print(
            "Old FAISS database removed."
        )

    except Exception as e:

        print(
            "Could not remove old FAISS database:",
            e
        )


# ============================================================
# STEP 14: CREATE OR LOAD FAISS
# ============================================================

if (

    os.path.exists(
        faiss_index_path
    )

    and

    os.path.exists(
        faiss_pickle_path
    )

    and

    os.path.exists(
        SIGNATURE_FILE
    )

):

    print(
        "\n" + "=" * 70
    )

    print(
        "LOADING EXISTING FAISS DATABASE"
    )

    print(
        "=" * 70
    )


    vector_db = FAISS.load_local(

        VECTOR_DB_PATH,

        embeddings,

        allow_dangerous_deserialization=True
    )


    print(
        "FAISS vector database loaded successfully!"
    )


else:

    print(
        "\n" + "=" * 70
    )

    print(
        "CREATING NEW FAISS DATABASE"
    )

    print(
        "=" * 70
    )


    vector_db = FAISS.from_documents(

        documents,

        embeddings
    )


    os.makedirs(
        VECTOR_DB_PATH,
        exist_ok=True
    )


    vector_db.save_local(
        VECTOR_DB_PATH
    )


    # Save dataset signature

    with open(
        SIGNATURE_FILE,
        "w"
    ) as file:

        file.write(
            dataset_signature
        )


    print(
        "FAISS vector database created successfully!"
    )

    print(
        "Number of indexed documents:",
        len(documents)
    )

    print(
        "Saved at:",
        VECTOR_DB_PATH
    )


# ============================================================
# STEP 15: SOURCE DETECTION
# ============================================================

SOURCE_ALIASES = {

    "Met Museum":
        [
            "met museum",
            "metropolitan museum",
            "the met",
            "met"
        ],

    "Smithsonian":
        [
            "smithsonian",
            "smithsonian museum"
        ],

    "Art Institute of Chicago":
        [
            "art institute of chicago",
            "chicago museum",
            "chicago"
        ],

    "Cleveland Museum":
        [
            "cleveland museum",
            "cleveland museum of art",
            "cleveland"
        ],

    "Victoria and Albert Museum":
        [
            "victoria and albert",
            "v&a",
            "v & a",
            "victoria albert",
            "v and a"
        ]
}


# ============================================================
# STEP 16: MOVEMENT DETECTION
# ============================================================

MOVEMENTS = [

    "Baroque",
    "Abstract",
    "Realism",
    "Expressionism",
    "Pop Art",
    "Conceptual",
    "Surrealism",
    "Impressionism",
    "Geometric Abstraction",
    "Minimalism",
    "Abstract Expressionism",
    "Feminist Art",
    "Traditional",
    "Post-Minimalism"
]


# ============================================================
# STEP 17: PERIOD DETECTION
# ============================================================

PERIODS = [

    "Contemporary",
    "Post-War",
    "Modern",
    "19th Century"
]


# ============================================================
# STEP 18: SEMANTIC SEARCH FUNCTION
# ============================================================

def search_artworks(
    query,
    k=5
):

    """
    Search the final fused artwork dataset.

    The function first checks for exact
    source/movement/period filters and then
    falls back to semantic similarity search.
    """

    query_lower = query.lower().strip()


    # ========================================================
    # SOURCE SEARCH
    # ========================================================

    detected_source = None


    for source, aliases in SOURCE_ALIASES.items():

        for alias in aliases:

            if alias in query_lower:

                detected_source = source

                break

        if detected_source:

            break


    if detected_source:

        print(
            f"\nDetected source filter: {detected_source}"
        )

        results = vector_db.similarity_search(

            query,

            k=k,

            filter={
                "source":
                    detected_source
            }
        )

        if results:

            return results


    # ========================================================
    # MOVEMENT SEARCH
    # ========================================================

    detected_movement = None


    for movement in MOVEMENTS:

        if movement.lower() in query_lower:

            detected_movement = movement

            break


    if detected_movement:

        print(
            f"\nDetected movement filter: {detected_movement}"
        )

        results = vector_db.similarity_search(

            query,

            k=k,

            filter={
                "movement":
                    detected_movement
            }
        )

        if results:

            return results


    # ========================================================
    # PERIOD SEARCH
    # ========================================================

    detected_period = None


    for period in PERIODS:

        if period.lower() in query_lower:

            detected_period = period

            break


    if detected_period:

        print(
            f"\nDetected period filter: {detected_period}"
        )

        results = vector_db.similarity_search(

            query,

            k=k,

            filter={
                "period":
                    detected_period
            }
        )

        if results:

            return results


    # ========================================================
    # NORMAL SEMANTIC SEARCH
    # ========================================================

    print(
        "\nUsing semantic similarity search..."
    )

    results = vector_db.similarity_search(

        query,

        k=k
    )


    return results


# ============================================================
# STEP 19: LOAD GOOGLE API KEY
# ============================================================

print(
    "\nLoading environment variables..."
)

load_dotenv()

api_key = os.getenv(
    "GOOGLE_API_KEY"
)


if not api_key:

    raise ValueError(

        "GOOGLE_API_KEY not found.\n"

        "Please make sure your .env file contains:\n"

        "GOOGLE_API_KEY=your_api_key"
    )


print(
    "Google API key loaded successfully!"
)


# ============================================================
# STEP 20: INITIALIZE GEMINI
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "LOADING GEMINI LLM"
)

print(
    "=" * 70
)


llm = ChatGoogleGenerativeAI(

    model="gemini-3.6-flash",

    google_api_key=api_key
)


print(
    "Gemini LLM loaded successfully!"
)


# ============================================================
# STEP 21: COMPLETE RAG FUNCTION
# ============================================================

def ask_artsense(
    question,
    k=5
):

    """
    Complete ArtSense AI RAG function.

    Question
        ↓
    Source / movement / period detection
        ↓
    FAISS retrieval
        ↓
    Artwork context
        ↓
    Gemini
        ↓
    Final answer
    """


    if not question:

        return (
            "Please enter a question about artworks."
        )


    question = str(
        question
    ).strip()


    # ========================================================
    # RETRIEVE DOCUMENTS
    # ========================================================

    results = search_artworks(

        question,

        k=k
    )


    if not results:

        return (

            "I could not find enough information "
            "in the artwork dataset."
        )


    # ========================================================
    # CREATE CONTEXT
    # ========================================================

    context_parts = []


    for i, doc in enumerate(results):

        context_parts.append(

            f"""
ARTWORK {i + 1}

{doc.page_content}
"""
        )


    context = "\n".join(
        context_parts
    )


    # ========================================================
    # RAG PROMPT
    # ========================================================

    prompt = f"""
You are ArtSense AI, an intelligent AI assistant
specialized in artwork discovery, recommendation,
and artwork analysis.

You are working with a fused artwork database
containing records from the original dataset and
multiple museum APIs.

Answer the user's question using ONLY the artwork
information provided in the context.

IMPORTANT RULES:

1. Do not invent artwork information.

2. Do not use outside knowledge.

3. Only use information available in the context.

4. If the requested information is not available,
   say:

   "I could not find enough information in the artwork dataset."

5. Mention artwork titles and artists whenever relevant.

6. When recommending artworks, explain briefly
   why each artwork matches the user's request.

7. When the user asks about a museum or data source,
   clearly mention the source.

8. When the user asks about art movement, use the
   Art Movement field.

9. When the user asks about art period, use the
   Art Period field.

10. When the user asks about price, use only the
    Price field provided in the context.

11. When comparing artworks, only compare information
    that exists in the context.

12. Do not assume that "Unknown" means a specific value.

13. Keep answers clear and easy to understand.

14. Use bullet points when listing multiple artworks.

15. Do not say that you searched the internet.

ARTWORK DATABASE CONTEXT:

{context}


USER QUESTION:

{question}


ARTSENSE AI ANSWER:
"""


    # ========================================================
    # GEMINI RESPONSE
    # ========================================================

    response = llm.invoke(
        prompt
    )


    answer = getattr(
        response,
        "text",
        None
    )


    if not answer:

        answer = str(
            response.content
        )


    return answer


# ============================================================
# STEP 22: TEST COMPLETE SYSTEM
# ============================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 70
    )

    print(
        "TESTING COMPLETE ARTSENSE AI RAG SYSTEM"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # TEST 1
    # --------------------------------------------------------

    question = (
        "Find artworks from the Met Museum."
    )


    print(
        "\nUSER QUESTION:"
    )

    print(
        question
    )


    print(
        "\nARTSENSE AI ANSWER:\n"
    )


    answer = ask_artsense(
        question
    )


    print(
        answer
    )


    # --------------------------------------------------------
    # ADDITIONAL TESTS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RAG PIPELINE TEST COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "\nFinal dataset records:",
        len(df)
    )

    print(
        "FAISS documents:",
        len(documents)
    )

    print(
        "Dataset sources:",
        df["source"].nunique()
    )

    print(
        "\nAvailable sources:"
    )

    for source in sorted(
        df["source"].unique()
    ):

        print(
            " -",
            source
        )