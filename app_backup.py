import streamlit as st
import speech_recognition as sr
import io
import pandas as pd
import os
import tempfile

from ocr_processor import process_ocr


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="ArtSense AI",
    page_icon="🎨",
    layout="wide"
)


# ==================================================
# IMPORT RAG FUNCTION
# ==================================================

from rag_pipeline import ask_artsense


# ==================================================
# SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "voice_processed" not in st.session_state:
    st.session_state.voice_processed = False


# ==================================================
# SPEECH TO TEXT
# ==================================================

def speech_to_text(audio_file):

    recognizer = sr.Recognizer()

    try:

        # Get audio bytes from Streamlit
        audio_bytes = audio_file.getvalue()

        # Convert to byte stream
        audio_stream = io.BytesIO(audio_bytes)

        # Read audio
        with sr.AudioFile(audio_stream) as source:

            audio_data = recognizer.record(source)

        # Convert speech to text
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


# ==================================================
# ASK AI FUNCTION
# ==================================================

def process_question(question):

    if not question or not question.strip():
        return

    question = question.strip()

    # Add user message
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


        # Add AI answer
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


    except Exception as e:

        error_message = f"""
Sorry, an error occurred while processing your question.

Error: {e}
"""

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message
            }
        )


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("🎨 ArtSense AI")

    st.write(
        "AI-powered artwork discovery and analysis system."
    )

    st.divider()


    # ----------------------------------------------
    # DATASET UPLOAD
    # ----------------------------------------------

    st.subheader("📤 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload artwork dataset",
        type=["csv"],
        help="Upload a CSV file containing artwork information."
    )


    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(
                uploaded_file
            )

            st.success(
                "Dataset uploaded successfully!"
            )

            st.write(
                f"Rows: {uploaded_df.shape[0]}"
            )

            st.write(
                f"Columns: {uploaded_df.shape[1]}"
            )


            with st.expander("Preview Dataset"):

                st.dataframe(
                    uploaded_df.head(),
                    use_container_width=True
                )


        except Exception as e:

            st.error(
                f"Could not read dataset: {e}"
            )


    st.divider()


    # ----------------------------------------------
    # OCR IMAGE UPLOAD
    # ----------------------------------------------

    st.subheader("🖼️ OCR Scanner")

    st.write(
        "Upload an artwork label, certificate, "
        "painting information card, or document."
    )


    uploaded_image = st.file_uploader(
        "Upload image for OCR",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        key="ocr_image"
    )


    if uploaded_image is not None:

        # Show uploaded image
        st.image(
            uploaded_image,
            caption="Uploaded Image",
            use_container_width=True
        )


        # Extract OCR button
        if st.button(
            "🔍 Extract Text",
            use_container_width=True,
            key="ocr_button"
        ):

            temp_path = None

            try:

                with st.spinner(
                    "Extracting text from image..."
                ):

                    # Get file extension
                    file_extension = os.path.splitext(
                        uploaded_image.name
                    )[1]


                    # Create temporary image file
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=file_extension
                    ) as temp_file:

                        temp_file.write(
                            uploaded_image.getvalue()
                        )

                        temp_path = temp_file.name


                    # Process OCR
                    extracted_text, output_file = process_ocr(
                        temp_path,
                        uploaded_image.name
                    )


                # Show success message
                st.success(
                    "Text extracted successfully!"
                )


                # Show OCR result
                st.text_area(
                    "📝 Extracted Text",
                    extracted_text,
                    height=250
                )


                # Show save location
                st.caption(
                    f"Saved to: {output_file}"
                )


            except Exception as e:

                st.error(
                    f"OCR Error: {str(e)}"
                )


            finally:

                # Delete temporary file
                if (
                    temp_path is not None
                    and os.path.exists(temp_path)
                ):

                    os.remove(temp_path)


    st.divider()


    # ----------------------------------------------
    # EXAMPLE QUESTIONS
    # ----------------------------------------------

    st.subheader("💡 Example Questions")

    example_questions = [

        "Find Baroque artworks",

        "Show me Abstract artworks",

        "Recommend artworks from the Contemporary period",

        "Which artworks are in excellent condition?",

        "Find artworks created in 2021",

        "Recommend interesting artworks from the dataset",

        "Find artworks by Tommaso Ottieri",

        "What is the cheapest artwork in the dataset?"

    ]


    for example in example_questions:

        if st.button(
            example,
            key=f"example_{example}"
        ):

            process_question(example)

            st.rerun()


    st.divider()


    # ----------------------------------------------
    # CLEAR CHAT
    # ----------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ==================================================
# MAIN PAGE HEADER
# ==================================================

st.title("🎨 ArtSense AI")

st.subheader(
    "AI-Powered Artwork Discovery & Analysis System"
)

st.write(
    """
Ask questions about artworks, artists, art movements,
periods, prices, conditions, and other information
available in the artwork dataset.
"""
)


# ==================================================
# PROJECT FEATURES
# ==================================================

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


# ==================================================
# CHAT HISTORY
# ==================================================

st.markdown(
    "## 💬 Chat with ArtSense AI"
)


# Show welcome message only when chat is empty
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

You can upload an artwork label or document
in the sidebar and extract text using OCR! 🖼️
"""
        )


# ==================================================
# DISPLAY CHAT MESSAGES
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(
            message["content"]
        )


# ==================================================
# TEXT CHAT INPUT
# ==================================================

user_question = st.chat_input(
    "Ask ArtSense AI about artworks..."
)


if user_question:

    process_question(
        user_question
    )

    st.rerun()


# ==================================================
# VOICE INPUT
# ==================================================

st.markdown(
    "### 🎤 Ask by Voice"
)


audio_file = st.audio_input(
    "Record your question"
)


if audio_file is not None:

    st.audio(
        audio_file
    )


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


        # Error occurred
        if error is not None:

            st.error(
                error
            )


        # Speech converted successfully
        elif speech_text:

            st.success(
                f"🎤 You said: {speech_text}"
            )


            process_question(
                speech_text
            )


            st.rerun()


# ==================================================
# FOOTER
# ==================================================

st.divider()


st.caption(
    "🎨 ArtSense AI | "
    "RAG + FAISS + HuggingFace Embeddings + "
    "Gemini LLM + Speech Recognition + OCR"
)