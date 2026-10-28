import streamlit as st
import speech_recognition as sr
import io
import pandas as pd
import os
import tempfile
import glob

from ocr_processor import process_ocr
from rag_pipeline import ask_artsense


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ArtSense AI",
    page_icon="🎨",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FINAL_DATASET = os.path.join(
    BASE_DIR,
    "fused_data",
    "final_fused_dataset.csv"
)

EDA_FOLDER = os.path.join(
    BASE_DIR,
    "fused_data",
    "eda_outputs"
)

API_FOLDER = os.path.join(
    BASE_DIR,
    "api_data"
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None

if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""


# ============================================================
# SPEECH TO TEXT
# ============================================================

def speech_to_text(audio_file):

    recognizer = sr.Recognizer()

    try:

        audio_bytes = audio_file.getvalue()

        audio_stream = io.BytesIO(audio_bytes)

        with sr.AudioFile(audio_stream) as source:

            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(
            audio_data,
            language="en-IN"
        )

        return text, None

    except sr.UnknownValueError:

        return None, (
            "Sorry, I could not understand the audio. "
            "Please speak clearly and try again."
        )

    except sr.RequestError as e:

        return None, (
            f"Speech recognition service error: {e}"
        )

    except Exception as e:

        return None, (
            f"Speech recognition error: {str(e)}"
        )


# ============================================================
# ASK AI
# ============================================================

def process_question(question):

    if not question or not question.strip():
        return

    question = question.strip()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    try:

        with st.spinner(
            "🎨 ArtSense AI is analyzing artworks..."
        ):

            answer = ask_artsense(question)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    except Exception as e:

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content":
                    f"Sorry, an error occurred.\n\nError: {e}"
            }
        )


# ============================================================
# LOAD FINAL FUSED DATASET
# ============================================================

@st.cache_data
def load_final_dataset():

    if not os.path.exists(FINAL_DATASET):
        return None

    try:

        df = pd.read_csv(FINAL_DATASET)

        return df

    except Exception:

        return None


final_df = load_final_dataset()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎨 ArtSense AI")

    st.write(
        "AI-powered artwork discovery and analysis system."
    )

    st.divider()

    # ========================================================
    # NAVIGATION
    # ========================================================

    st.subheader("📂 Navigation")

    page = st.radio(
        "Go to",
        [
            "🏠 Chat",
            "🔍 Artwork Explorer",
            "📊 EDA Dashboard",
            "🖼️ OCR Scanner",
            "📤 Dataset Upload",
            "🌐 API Sources"
        ],
        key="navigation"
    )

    st.divider()

    # ========================================================
    # QUICK DATASET INFO
    # ========================================================

    st.subheader("📚 Dataset")

    if final_df is not None:

        st.metric(
            "Total Records",
            len(final_df)
        )

        st.metric(
            "Attributes",
            len(final_df.columns)
        )

    else:

        st.warning(
            "Final fused dataset not found."
        )

    st.divider()

    # ========================================================
    # EXAMPLE QUESTIONS
    # ========================================================

    if page == "🏠 Chat":

        st.subheader("💡 Example Questions")

        example_questions = [

            "Find Baroque artworks",

            "Show me Abstract artworks",

            "Recommend Contemporary artworks",

            "Which artworks are in excellent condition?",

            "Find artworks created in 2021",

            "Recommend interesting artworks",

            "Find artworks by Tommaso Ottieri",

            "What is the cheapest artwork?"
        ]

        for example in example_questions:

            if st.button(
                example,
                key=f"example_{example}",
                use_container_width=True
            ):

                process_question(example)

                st.rerun()

        st.divider()

    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# PAGE 1 - CHAT
# ============================================================

if page == "🏠 Chat":

    st.title("🎨 ArtSense AI")

    st.subheader(
        "AI-Powered Artwork Discovery & Analysis System"
    )

    st.write(
        """
        Ask questions about artworks, artists, art movements,
        periods, prices, conditions and other information
        available in the artwork dataset.
        """
    )

    # ========================================================
    # FEATURES
    # ========================================================

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.info(
            "📚 **RAG**\n\n"
            "Retrieves relevant artwork information."
        )

    with col2:

        st.info(
            "🔍 **Semantic Search**\n\n"
            "Searches artworks by meaning."
        )

    with col3:

        st.info(
            "🤖 **Gemini LLM**\n\n"
            "Generates intelligent answers."
        )

    with col4:

        st.info(
            "🎤 **Voice Input**\n\n"
            "Ask questions using your voice."
        )

    with col5:

        st.info(
            "🖼️ **OCR Scanner**\n\n"
            "Extracts text from artwork images."
        )

    st.divider()

    # ========================================================
    # CHAT
    # ========================================================

    st.markdown("## 💬 Chat with ArtSense AI")

    if len(st.session_state.messages) == 0:

        with st.chat_message("assistant"):

            st.write(
                """
                👋 Hello! I am **ArtSense AI**.

                You can ask me questions such as:

                - Find Baroque artworks
                - Show Abstract paintings
                - Which artwork is the cheapest?
                - Recommend Contemporary artworks
                - Find artworks in excellent condition
                - Find artworks by a particular artist

                You can also use the microphone below! 🎤
                """
            )

    # ========================================================
    # CHAT HISTORY
    # ========================================================

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.write(message["content"])

    # ========================================================
    # TEXT INPUT
    # ========================================================

    user_question = st.chat_input(
        "Ask ArtSense AI about artworks..."
    )

    if user_question:

        process_question(user_question)

        st.rerun()

    # ========================================================
    # SPEECH INPUT
    # ========================================================

    st.markdown("### 🎤 Ask by Voice")

    audio_file = st.audio_input(
        "Record your question"
    )

    if audio_file is not None:

        st.audio(audio_file)

        if st.button(
            "🎙️ Ask ArtSense AI",
            key="voice_button"
        ):

            with st.spinner(
                "Converting your speech to text..."
            ):

                speech_text, error = speech_to_text(
                    audio_file
                )

            if error is not None:

                st.error(error)

            elif speech_text:

                st.success(
                    f"🎤 You said: {speech_text}"
                )

                process_question(
                    speech_text
                )

                st.rerun()


# ============================================================
# PAGE 2 - ARTWORK EXPLORER
# ============================================================

elif page == "🔍 Artwork Explorer":

    st.title("🔍 Artwork Explorer")

    st.write(
        "Explore the final fused artwork dataset."
    )

    if final_df is None:

        st.error(
            "Final fused dataset could not be found."
        )

        st.info(
            "Run: python final_fusion.py"
        )

    else:

        # ====================================================
        # DATASET METRICS
        # ====================================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Artworks",
                len(final_df)
            )

        with col2:

            st.metric(
                "Attributes",
                len(final_df.columns)
            )

        with col3:

            if "source" in final_df.columns:

                st.metric(
                    "Data Sources",
                    final_df["source"].nunique()
                )

        with col4:

            if "artist" in final_df.columns:

                st.metric(
                    "Artists",
                    final_df["artist"].nunique()
                )

        st.divider()

        # ====================================================
        # FILTERS
        # ====================================================

        st.subheader("🎯 Filter Artworks")

        filter_col1, filter_col2, filter_col3 = st.columns(3)

        filtered_df = final_df.copy()

        with filter_col1:

            if "source" in final_df.columns:

                sources = sorted(
                    final_df["source"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                selected_source = st.multiselect(
                    "Data Source",
                    sources
                )

                if selected_source:

                    filtered_df = filtered_df[
                        filtered_df["source"].isin(
                            selected_source
                        )
                    ]

        with filter_col2:

            if "movement" in final_df.columns:

                movements = sorted(
                    final_df["movement"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                selected_movement = st.multiselect(
                    "Art Movement",
                    movements
                )

                if selected_movement:

                    filtered_df = filtered_df[
                        filtered_df["movement"].isin(
                            selected_movement
                        )
                    ]

        with filter_col3:

            if "period" in final_df.columns:

                periods = sorted(
                    final_df["period"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                selected_period = st.multiselect(
                    "Art Period",
                    periods
                )

                if selected_period:

                    filtered_df = filtered_df[
                        filtered_df["period"].isin(
                            selected_period
                        )
                    ]

        # ====================================================
        # SEARCH
        # ====================================================

        search_text = st.text_input(
            "🔎 Search by title, artist or description",
            placeholder="Example: Monet, Baroque, portrait..."
        )

        if search_text:

            search_text = search_text.lower()

            searchable_columns = [
                "title",
                "artist",
                "description",
                "medium",
                "classification"
            ]

            mask = pd.Series(
                False,
                index=filtered_df.index
            )

            for column in searchable_columns:

                if column in filtered_df.columns:

                    mask = mask | (
                        filtered_df[column]
                        .fillna("")
                        .astype(str)
                        .str.lower()
                        .str.contains(
                            search_text,
                            na=False
                        )
                    )

            filtered_df = filtered_df[mask]

        st.divider()

        st.write(
            f"Showing **{len(filtered_df)}** artworks"
        )

        # ====================================================
        # DISPLAY DATA
        # ====================================================

        display_columns = [
            "title",
            "artist",
            "date",
            "medium",
            "source",
            "movement",
            "period",
            "description"
        ]

        display_columns = [
            column
            for column in display_columns
            if column in filtered_df.columns
        ]

        st.dataframe(
            filtered_df[display_columns],
            use_container_width=True,
            height=500
        )

        # ====================================================
        # DOWNLOAD
        # ====================================================

        csv_data = filtered_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Filtered Dataset",
            csv_data,
            "artsense_filtered_dataset.csv",
            "text/csv",
            use_container_width=True
        )


# ============================================================
# PAGE 3 - EDA DASHBOARD
# ============================================================

elif page == "📊 EDA Dashboard":

    st.title("📊 Exploratory Data Analysis")

    st.write(
        "Visual analysis of the final fused ArtSense AI dataset."
    )

    if not os.path.exists(EDA_FOLDER):

        st.error(
            "EDA output folder not found."
        )

        st.info(
            "Run: python eda.py"
        )

    else:

        # ====================================================
        # DATASET SUMMARY
        # ====================================================

        if final_df is not None:

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Records",
                    len(final_df)
                )

            with col2:

                st.metric(
                    "Attributes",
                    len(final_df.columns)
                )

            with col3:

                if "record_source" in final_df.columns:

                    api_count = (
                        final_df["record_source"]
                        .astype(str)
                        .str.contains(
                            "API",
                            case=False,
                            na=False
                        )
                        .sum()
                    )

                    st.metric(
                        "API Records",
                        api_count
                    )

            with col4:

                if "price_numeric" in final_df.columns:

                    valid_prices = pd.to_numeric(
                        final_df["price_numeric"],
                        errors="coerce"
                    ).dropna()

                    st.metric(
                        "Valid Prices",
                        len(valid_prices)
                    )

        st.divider()

        # ====================================================
        # GRAPH HELPER
        # ====================================================

        def show_graph(filename, title):

            path = os.path.join(
                EDA_FOLDER,
                filename
            )

            if os.path.exists(path):

                st.subheader(title)

                st.image(
                    path,
                    use_container_width=True
                )

        # ====================================================
        # SOURCE DISTRIBUTION
        # ====================================================

        show_graph(
            "01_source_distribution.png",
            "1️⃣ Data Source Distribution"
        )

        # ====================================================
        # ORIGINAL VS API
        # ====================================================

        show_graph(
            "02_original_vs_api.png",
            "2️⃣ Original Dataset vs API Dataset"
        )

        # ====================================================
        # ART MOVEMENT
        # ====================================================

        show_graph(
            "03_art_movement_distribution.png",
            "3️⃣ Art Movement Distribution"
        )

        # ====================================================
        # ART PERIOD
        # ====================================================

        show_graph(
            "04_art_period_distribution.png",
            "4️⃣ Art Period Distribution"
        )

        # ====================================================
        # TOP ARTISTS
        # ====================================================

        show_graph(
            "05_top_artists.png",
            "5️⃣ Top Artists"
        )

        # ====================================================
        # MEDIUM
        # ====================================================

        show_graph(
            "06_medium_distribution.png",
            "6️⃣ Medium Distribution"
        )

        # ====================================================
        # PRICE DISTRIBUTION
        # ====================================================

        show_graph(
            "07_price_distribution.png",
            "7️⃣ Price Distribution"
        )

        # ====================================================
        # PRICE BOXPLOT
        # ====================================================

        show_graph(
            "08_price_boxplot.png",
            "8️⃣ Price Boxplot"
        )

        # ====================================================
        # EXPENSIVE ARTWORKS
        # ====================================================

        show_graph(
            "09_top_10_expensive.png",
            "9️⃣ Top 10 Most Expensive Artworks"
        )

        # ====================================================
        # SOURCE VS MEDIUM
        # ====================================================

        show_graph(
            "10_source_vs_medium.png",
            "🔟 Data Source vs Medium"
        )

        # ====================================================
        # SOURCE VS PERIOD
        # ====================================================

        show_graph(
            "11_source_vs_period.png",
            "1️⃣1️⃣ Data Source vs Art Period"
        )

        # ====================================================
        # CORRELATION
        # ====================================================

        show_graph(
            "12_correlation_heatmap.png",
            "1️⃣2️⃣ Correlation Heatmap"
        )

        # ====================================================
        # API COMPLETENESS
        # ====================================================

        show_graph(
            "13_api_completeness.png",
            "1️⃣3️⃣ API Data Completeness"
        )

        # ====================================================
        # SUMMARY CSV
        # ====================================================

        summary_file = os.path.join(
            EDA_FOLDER,
            "eda_summary.csv"
        )

        if os.path.exists(summary_file):

            st.divider()

            st.subheader(
                "📄 EDA Summary Data"
            )

            try:

                summary_df = pd.read_csv(
                    summary_file
                )

                st.dataframe(
                    summary_df,
                    use_container_width=True
                )

            except Exception:

                st.warning(
                    "Could not read EDA summary."
                )


# ============================================================
# PAGE 4 - OCR SCANNER
# ============================================================

elif page == "🖼️ OCR Scanner":

    st.title("🖼️ OCR Scanner")

    st.write(
        """
        Upload an artwork label, certificate,
        painting information card, or document.
        """
    )

    uploaded_image = st.file_uploader(
        "Upload image for OCR",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        key="ocr_page_image"
    )

    if uploaded_image is not None:

        st.image(
            uploaded_image,
            caption="Uploaded Artwork Image",
            use_container_width=True
        )

        if st.button(
            "🔍 Extract Text",
            use_container_width=True
        ):

            temp_path = None

            try:

                with st.spinner(
                    "Extracting text from image..."
                ):

                    file_extension = os.path.splitext(
                        uploaded_image.name
                    )[1]

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=file_extension
                    ) as temp_file:

                        temp_file.write(
                            uploaded_image.getvalue()
                        )

                        temp_path = temp_file.name

                    extracted_text, output_file = process_ocr(
                        temp_path,
                        uploaded_image.name
                    )

                st.success(
                    "Text extracted successfully!"
                )

                st.session_state.ocr_text = extracted_text

                st.text_area(
                    "📝 Extracted Text",
                    extracted_text,
                    height=250
                )

                st.caption(
                    f"Saved to: {output_file}"
                )

                # --------------------------------------------
                # SEND OCR TEXT TO AI
                # --------------------------------------------

                if extracted_text:

                    if st.button(
                        "🤖 Ask ArtSense AI About This Text",
                        use_container_width=True
                    ):

                        process_question(
                            extracted_text
                        )

                        st.success(
                            "OCR text sent to ArtSense AI."
                        )

            except Exception as e:

                st.error(
                    f"OCR Error: {str(e)}"
                )

            finally:

                if (
                    temp_path is not None
                    and os.path.exists(temp_path)
                ):

                    os.remove(temp_path)


# ============================================================
# PAGE 5 - DATASET UPLOAD
# ============================================================

elif page == "📤 Dataset Upload":

    st.title("📤 Dataset Upload")

    st.write(
        """
        Upload a CSV artwork dataset to inspect its
        structure and contents.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload artwork dataset",
        type=["csv"],
        help="Upload a CSV file containing artwork information.",
        key="dataset_page_upload"
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(
                uploaded_file
            )

            st.session_state.uploaded_df = uploaded_df

            st.success(
                "Dataset uploaded successfully!"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Rows",
                    uploaded_df.shape[0]
                )

            with col2:

                st.metric(
                    "Columns",
                    uploaded_df.shape[1]
                )

            with col3:

                st.metric(
                    "Missing Values",
                    int(
                        uploaded_df.isnull()
                        .sum()
                        .sum()
                    )
                )

            st.divider()

            st.subheader(
                "📋 Dataset Preview"
            )

            st.dataframe(
                uploaded_df.head(20),
                use_container_width=True
            )

            st.subheader(
                "📊 Dataset Information"
            )

            info_df = pd.DataFrame(
                {
                    "Column":
                        uploaded_df.columns,
                    "Data Type":
                        [
                            str(dtype)
                            for dtype
                            in uploaded_df.dtypes
                        ],
                    "Missing Values":
                        [
                            int(value)
                            for value
                            in uploaded_df.isnull()
                            .sum()
                            .values
                        ]
                }
            )

            st.dataframe(
                info_df,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Could not read dataset: {e}"
            )

    else:

        st.info(
            "Upload a CSV file to begin."
        )

    st.divider()

    st.subheader(
        "📌 Current Final Dataset"
    )

    if final_df is not None:

        st.write(
            f"The current ArtSense AI fused dataset contains "
            f"**{len(final_df)} records** and "
            f"**{len(final_df.columns)} attributes**."
        )

        st.dataframe(
            final_df.head(10),
            use_container_width=True
        )


# ============================================================
# PAGE 6 - API SOURCES
# ============================================================

elif page == "🌐 API Sources":

    st.title("🌐 API Data Sources")

    st.write(
        """
        ArtSense AI integrates artwork information
        from five external museum APIs.
        """
    )

    # ========================================================
    # API CARDS
    # ========================================================

    api_information = [

        (
            "🏛️ The Metropolitan Museum of Art",
            "Met Museum",
            "Provides artwork metadata such as title, "
            "artist, date, medium, department and image information."
        ),

        (
            "🏛️ Smithsonian",
            "Smithsonian",
            "Provides museum collection information "
            "and artwork metadata."
        ),

        (
            "🎨 Art Institute of Chicago",
            "Art Institute of Chicago",
            "Provides artwork records including title, "
            "artist, date, medium and collection information."
        ),

        (
            "🏛️ Cleveland Museum of Art",
            "Cleveland Museum",
            "Provides artwork metadata including "
            "artist, date, medium and collection details."
        ),

        (
            "🏛️ Victoria and Albert Museum",
            "Victoria and Albert Museum",
            "Provides collection information including "
            "artwork titles, creators, dates and materials."
        )
    ]

    for title, source_name, description in api_information:

        with st.container(border=True):

            st.subheader(title)

            st.write(
                f"**Source:** {source_name}"
            )

            st.write(
                description
            )

    st.divider()

    # ========================================================
    # API DATA FILES
    # ========================================================

    st.subheader(
        "📁 Downloaded API Datasets"
    )

    if os.path.exists(API_FOLDER):

        api_files = glob.glob(
            os.path.join(
                API_FOLDER,
                "*.csv"
            )
        )

        if api_files:

            api_table = []

            for file_path in api_files:

                try:

                    temp_df = pd.read_csv(
                        file_path
                    )

                    api_table.append(
                        {
                            "File":
                                os.path.basename(
                                    file_path
                                ),
                            "Records":
                                len(temp_df),
                            "Attributes":
                                len(temp_df.columns)
                        }
                    )

                except Exception:

                    pass

            if api_table:

                st.dataframe(
                    pd.DataFrame(api_table),
                    use_container_width=True
                )

        else:

            st.info(
                "No API CSV files found."
            )

    else:

        st.warning(
            "api_data folder not found."
        )

    # ========================================================
    # FINAL FUSION
    # ========================================================

    st.divider()

    st.subheader(
        "🔗 Final Data Integration"
    )

    st.write(
        """
        The five API datasets are standardized and combined
        with the original artwork dataset.
        """
    )

    if final_df is not None:

        st.success(
            f"Final fused dataset: "
            f"{len(final_df)} records × "
            f"{len(final_df.columns)} attributes"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎨 ArtSense AI | "
    "5 Museum APIs + Data Fusion + EDA + "
    "RAG + FAISS + HuggingFace Embeddings + "
    "Gemini LLM + Speech Recognition + OCR"
)