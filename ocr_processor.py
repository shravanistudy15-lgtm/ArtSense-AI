import os
import requests
import pandas as pd
from dotenv import load_dotenv


# ==================================================
# LOAD API KEY
# ==================================================

load_dotenv()

OCR_API_KEY = os.getenv("OCR_SPACE_API_KEY")

OCR_URL = "https://api.ocr.space/parse/image"


# ==================================================
# EXTRACT TEXT USING OCR API
# ==================================================

def extract_text_from_image(image_path):

    # Open image
    with open(image_path, "rb") as image_file:

        response = requests.post(
            OCR_URL,

            files={
                "file": image_file
            },

            data={
                "apikey": OCR_API_KEY,
                "language": "eng",
                "OCREngine": 2
            }
        )


    # Convert response to JSON
    result = response.json()


    # Check for OCR error
    if result.get("IsErroredOnProcessing"):

        error_message = result.get(
            "ErrorMessage",
            "OCR processing failed."
        )

        return f"OCR Error: {error_message}"


    # Get parsed results
    parsed_results = result.get(
        "ParsedResults",
        []
    )


    # No text found
    if not parsed_results:

        return "No text detected in the image."


    # Extract text
    extracted_text = parsed_results[0].get(
        "ParsedText",
        ""
    )


    return extracted_text.strip()


# ==================================================
# SAVE OCR DATA
# ==================================================

def save_ocr_data(
    image_name,
    extracted_text
):

    os.makedirs(
        "api_data",
        exist_ok=True
    )


    ocr_data = pd.DataFrame({

        "source": [
            "OCR.Space"
        ],

        "file_name": [
            image_name
        ],

        "extracted_text": [
            extracted_text
        ]

    })


    output_file = (
        "api_data/ocr_extracted_data.csv"
    )


    # Append if file already exists
    if os.path.exists(output_file):

        ocr_data.to_csv(
            output_file,
            mode="a",
            header=False,
            index=False
        )

    else:

        ocr_data.to_csv(
            output_file,
            index=False
        )


    return output_file


# ==================================================
# COMPLETE OCR FUNCTION
# ==================================================

def process_ocr(
    image_path,
    image_name
):

    # Extract text
    extracted_text = (
        extract_text_from_image(
            image_path
        )
    )


    # Save OCR result
    output_file = save_ocr_data(
        image_name,
        extracted_text
    )


    return extracted_text, output_file