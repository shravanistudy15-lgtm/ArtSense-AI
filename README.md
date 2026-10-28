# ArtSense AI – AI-Powered Artwork Discovery System

## 1. Project Overview

ArtSense AI is an AI-powered artwork discovery application that uses Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and data from museum APIs to help users explore artwork information.

The project combines an original artwork dataset with additional metadata collected from five museum APIs. The resulting fused dataset is used for exploratory data analysis and to enhance the knowledge base of the RAG application.

## 2. Objectives

- Collect artwork information from multiple museum APIs.
- Combine the original dataset with additional API records.
- Perform data cleaning, quality assessment, and exploratory data analysis.
- Build a semantic retrieval system using embeddings and FAISS.
- Use Gemini to generate responses based on retrieved artwork information.
- Provide an interactive interface for artwork discovery through Streamlit.

## 3. Technologies Used

- **Programming Language:** Python
- **User Interface:** Streamlit
- **Data Processing:** Pandas
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database:** FAISS
- **Language Model:** Gemini
- **RAG Framework:** LangChain and related integrations
- **Data Visualization:** Matplotlib and Seaborn
- **Data Storage:** CSV and Excel files

## 4. Data Sources

The project uses an original artwork dataset and data collected from the following museum APIs:

1. The Metropolitan Museum of Art
2. Smithsonian Institution
3. Art Institute of Chicago
4. Cleveland Museum of Art
5. Victoria and Albert Museum

The API records provide additional metadata such as artwork medium, department, classification, culture, description, image URL, and source information where available.

## 5. Dataset Information

| Dataset | Records | Attributes |
|---|---:|---:|
| Original dataset | 754 | 8 |
| Final fused dataset | 957 | 17 |
| Additional API records | 203 | Varies by source |

The final fused dataset contains the original records and additional API records. The datasets were combined through concatenation because no matching keys were found between the original and API datasets.

## 6. Project Structure

```text
ArtSense-AI/
│
├── app.py
├── rag_pipeline.py
├── api_fusion.py
├── final_fusion.py
├── eda.py
│
├── data/
│   └── artworks.csv.csv
│
├── api_data/
│   ├── met_artworks.csv
│   ├── smithsonian_artworks.csv
│   ├── chicago_artworks.csv
│   ├── cleveland_artworks.csv
│   └── va_artworks.csv
│
├── fused_data/
│   ├── final_fused_dataset.csv
│   ├── final_fused_dataset.xlsx
│   └── eda_outputs/
│
├── apis/
├── README.md
└── requirements.txt
```

*Note: This is the expected project structure based on the implemented workflow. Filenames and folders should be adjusted to match the files actually present in the project.*

## 7. Installation and Setup

### Prerequisites

- Python installed on the system
- Visual Studio Code or another Python IDE
- Internet connection for API access and language-model services
- A Gemini API key if required by the application

### Step 1: Open the Project

Open the `ArtSense-AI` folder in Visual Studio Code.

### Step 2: Create a Virtual Environment

Open the terminal and run:

```powershell
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

If the project contains a `requirements.txt` file, run:

```powershell
python -m pip install -r requirements.txt
```

If the file does not exist, install the dependencies specified by the project's imports and configuration before running the application.

### Step 4: Configure the API Key

Configure the Gemini API key using the environment variable or configuration method expected by `rag_pipeline.py`. Keep the key private and do not commit it to a public repository.

### Step 5: Run the Application

If the Streamlit entry-point file is `app.py`, run:

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

## 8. Data Processing Workflow

The project follows these main stages:

1. Collect artwork records from museum APIs.
2. Save the API responses as structured CSV datasets.
3. Standardize and combine the API datasets.
4. Combine the original dataset with the API records.
5. Perform data-quality checks and exploratory data analysis.
6. Convert artwork records into documents for the RAG pipeline.
7. Generate embeddings using the Sentence Transformers model.
8. Store the vectors in FAISS.
9. Retrieve relevant artwork documents for user questions.
10. Pass retrieved context to Gemini to generate responses.

## 9. RAG Architecture

```text
Original Dataset + Museum API Data
                |
                v
          Fused CSV Dataset
                |
                v
          Data Preparation
                |
                v
       Artwork Document Creation
                |
                v
       Sentence Transformer Embeddings
                |
                v
             FAISS Index
                |
                v
        Semantic Document Retrieval
                |
                v
       Retrieved Context + User Query
                |
                v
             Gemini LLM
                |
                v
          Generated Response
```

## 10. Example Query

A user can submit a question such as:

**Question:** Find artworks from the Met Museum.

The application searches the artwork knowledge base and returns relevant records, including available titles, artists, and museum source information.

The accuracy and completeness of each response depend on the available metadata and retrieval results.

## 11. Exploratory Data Analysis

The EDA stage examines the fused dataset through:

- Source distribution
- Art movement distribution
- Art period distribution
- Top artists
- Medium distribution
- Price distribution and boxplots
- Correlation analysis
- API data completeness

The generated visualizations are stored in `fused_data/eda_outputs/`, subject to the actual project folder structure.

## 12. Limitations

- Some API records contain missing or incomplete metadata.
- Different museums use different identifiers and metadata formats.
- The original and API datasets were concatenated rather than matched record by record.
- The system may return incomplete information when relevant metadata is missing or retrieval is unsuccessful.
- Additional testing is required to measure answer accuracy and compare the original and enhanced RAG systems.

## 13. Future Enhancements

- Improve metadata cleaning and artwork record matching.
- Improve retrieval accuracy and relevance.
- Add filters for museum, artist, medium, period, and movement.
- Improve artwork image presentation.
- Evaluate responses using a larger set of verified questions.
- Add more museum data sources and improve the user interface.

## 14. Conclusion

ArtSense AI combines data fusion, exploratory data analysis, semantic retrieval, and LLM-based response generation to support artwork discovery. By incorporating additional museum metadata into its knowledge base, the project provides a foundation for richer artwork searches and more informative responses.

## 15. API References

- The Metropolitan Museum of Art: https://metmuseum.github.io/
- Smithsonian Open Access: https://www.si.edu/openaccess
- Art Institute of Chicago API: https://api.artic.edu/docs/
- Cleveland Museum of Art API: https://openaccess-api.clevelandart.org/
- Victoria and Albert Museum: https://developers.vam.ac.uk/

