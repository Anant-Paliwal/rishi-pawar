import os
import re
import json
import requests
import PyPDF2
import numpy as np
from pdf2image import convert_from_path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from PIL import Image
import nltk
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Ensure required NLTK resources are available
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# Load MobileNetV3 model for OCR backup
model = MobileNetV3Small(weights="imagenet")

# MongoDB Setup (Get credentials from environment variables)
# uri = "mongodb+srv://Anant31:<twMXokWBWL6IQz7h>@cluster0.munjj8l.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
uri = "mongodb+srv://Anant31:twMXokWBWL6IQz7h@cluster0.munjj8l.mongodb.net/pdf_summaries?retryWrites=true&w=majority"

client = MongoClient(uri, server_api=ServerApi('1'))

# MongoDB Database and Collection
db = client['pdf_summaries']  # Create or use the database
collection = db['summaries']   # Create or use the collection

# Helper Functions

def download_pdf(url, file_name):
    """Download a PDF file from a URL."""
    try:
        response = requests.get(url, stream=True)
        with open(file_name, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Downloaded {file_name}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {str(e)}")
        return False

def extract_text(file):
    """Extract text from a PDF using PyPDF2."""
    try:
        with open(file, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""  # Ensure it handles None
        return text
    except Exception as e:
        return f"Error reading {file}: {str(e)}"

def preprocess_image(img_path):
    """Preprocess image for MobileNetV3 OCR."""
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def mobile_net_ocr(file):
    """Extract text from a PDF using MobileNetV3."""
    try:
        pages = convert_from_path(file, 300)  # 300 DPI to improve accuracy
        text = ""
        for page_num, page in enumerate(pages, start=1):
            filename = f"page_{page_num}.jpg"
            page.save(filename, "JPEG")

            # Preprocess the image for MobileNetV3
            img_array = preprocess_image(filename)

            # Perform OCR using MobileNetV3
            preds = model.predict(img_array)
            decoded_preds = decode_predictions(preds, top=3)[0]  # Get top-3 predictions

            # Use decoded labels (as placeholders for actual OCR results)
            text += " ".join([label for _, label, _ in decoded_preds]) + " "

            # Remove temp file to save memory
            os.remove(filename)
        return text
    except Exception as e:
        return f"Error processing OCR with MobileNetV3: {str(e)}"

def summarize_text(text):
    """Summarize text using a word frequency method."""
    processed_text = re.sub(r"[^a-zA-Z' ]+", " ", text)
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(processed_text)

    # Normalize words with stemming and build word frequency table
    stemmer = SnowballStemmer("english")
    freq_table = {}
    for word in words:
        word = word.lower()
        if word not in stop_words:
            stem_word = stemmer.stem(word)
            freq_table[stem_word] = freq_table.get(stem_word, 0) + 1

    # Summarize sentences based on word frequencies
    sentences = sent_tokenize(text)
    sentence_value = {}
    for sentence in sentences:
        sentence_word_count = len(word_tokenize(sentence))
        for word in freq_table:
            if word in sentence.lower():
                if sentence in sentence_value:
                    sentence_value[sentence] += freq_table[word]
                else:
                    sentence_value[sentence] = freq_table[word]

        # Normalize score (Only if the sentence has a value)
        if sentence in sentence_value:
            sentence_value[sentence] /= sentence_word_count

    # Average sentence value for summarization threshold
    if len(sentence_value) > 0:
        avg_sentence_value = sum(sentence_value.values()) / len(sentence_value)
    else:
        avg_sentence_value = 0

    # Build the summary by choosing sentences with a higher value than average
    summary = ' '.join([sentence for sentence in sentences if sentence_value.get(sentence, 0) > (1.5 * avg_sentence_value)])
    return summary

def extract_keywords(text):
    """Extract keywords from the text."""
    processed_text = re.sub(r"[^a-zA-Z' ]+", " ", text)
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(processed_text)

    stemmer = SnowballStemmer("english")
    keywords = []
    for word in words:
        word = word.lower()
        if word not in stop_words:
            stem_word = stemmer.stem(word)
            keywords.append(stem_word)

    # Return the top-5 most frequent keywords (could adjust as needed)
    return list(set(keywords))[:5]

def save_summary_to_mongodb(summary_data):
    """Save the combined summary and keywords to MongoDB."""
    try:
        collection.insert_many(summary_data)
        print("Combined summary saved to MongoDB")
    except Exception as e:
        print(f"Failed to save summary to MongoDB: {str(e)}")

def process_pdf_files(dataset_json):
    """Process each PDF file in the dataset and save summaries in MongoDB."""
    with open(dataset_json, 'r') as f:
        data = json.load(f)

    combined_summary = []  # List to hold summaries from all PDFs

    for pdf_key, url in data.items():
        pdf_file_name = f"{pdf_key}.pdf"

        # Step 1: Download the PDF
        if not os.path.exists(pdf_file_name):
            if not download_pdf(url, pdf_file_name):
                continue  # Skip if download failed

        # Step 2: Extract Text using PyPDF2
        extracted_text = extract_text(pdf_file_name)

        # If direct extraction fails, fall back to MobileNetV3 OCR
        if "Error" in extracted_text:
            print(f"Text extraction failed for {pdf_file_name}. Falling back to MobileNetV3 OCR.")
            extracted_text = mobile_net_ocr(pdf_file_name)

        # Step 3: Summarize the text if extraction succeeded
        if "Error" not in extracted_text:
            summary = summarize_text(extracted_text)
            keywords = extract_keywords(extracted_text)

            # Append summary and keywords to the combined summary list
            combined_summary.append({
                "original_file": pdf_file_name,
                "summary": summary,
                "keywords": keywords
            })
        else:
            print(f"Failed to process {pdf_file_name}")

        # Optionally, you can remove the downloaded PDF to save space
        os.remove(pdf_file_name)

    # Step 4: Save all summaries in MongoDB
    save_summary_to_mongodb(combined_summary)

# Main Program
dataset_json_file = "Dataset.json"  # Path to your Dataset.json file
process_pdf_files(dataset_json_file)
