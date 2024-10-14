# Import necessary libraries
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
from flask import Flask, render_template, jsonify, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import logging

# Download necessary NLTK resources
# These are required for text processing tasks
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# Load MobileNetV3 model for OCR backup
# This model will be used if standard text extraction fails
model = MobileNetV3Small(weights="imagenet")

# MongoDB Setup
# This establishes a connection to our MongoDB database
uri = "mongodb+srv://Anant31:twMXokWBWL6IQz7h@cluster0.munjj8l.mongodb.net/pdf_summaries?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['pdf_summaries']
collection = db['summaries']

# Flask app setup
# Initialize the Flask application and configure it
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PDF_FOLDER'] = 'pdfs'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Ensure necessary folders exist
# This creates the upload and PDF folders if they don't exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['PDF_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Set up logging
# This configures the logging system to help with debugging
logging.basicConfig(level=logging.DEBUG)

# Helper Functions

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def download_pdf(url, file_name):
    """
    Download a PDF file from a given URL and save it locally.
    """
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
    """
    Extract text from a PDF file using PyPDF2.
    """
    try:
        with open(file, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error reading {file}: {str(e)}"

def preprocess_image(img_path):
    """
    Preprocess an image for use with MobileNetV3 OCR.
    """
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def mobile_net_ocr(file):
    """
    Extract text from a PDF using MobileNetV3 when standard extraction fails.
    """
    try:
        pages = convert_from_path(file, 300)
        text = ""
        for page_num, page in enumerate(pages, start=1):
            filename = f"page_{page_num}.jpg"
            page.save(filename, "JPEG")
            img_array = preprocess_image(filename)
            preds = model.predict(img_array)
            decoded_preds = decode_predictions(preds, top=3)[0]
            text += " ".join([label for _, label, _ in decoded_preds]) + " "
            os.remove(filename)
        return text
    except Exception as e:
        return f"Error processing OCR with MobileNetV3: {str(e)}"

def summarize_text(text):
    """
    Generate a summary of the given text using a word frequency method.
    """
    processed_text = re.sub(r"[^a-zA-Z' ]+", " ", text)
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(processed_text)

    stemmer = SnowballStemmer("english")
    freq_table = {}
    for word in words:
        word = word.lower()
        if word not in stop_words:
            stem_word = stemmer.stem(word)
            freq_table[stem_word] = freq_table.get(stem_word, 0) + 1

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

        if sentence in sentence_value:
            sentence_value[sentence] /= sentence_word_count

    if len(sentence_value) > 0:
        avg_sentence_value = sum(sentence_value.values()) / len(sentence_value)
    else:
        avg_sentence_value = 0

    summary = ' '.join([sentence for sentence in sentences if sentence_value.get(sentence, 0) > (1.5 * avg_sentence_value)])
    return summary

def extract_keywords(text):
    """
    Extract the top 5 keywords from the given text.
    """
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

    return list(set(keywords))[:5]

def save_summary_to_mongodb(summary_data):
    """
    Save the combined summary and keywords to MongoDB.
    """
    try:
        collection.insert_many(summary_data)
        print("Combined summary saved to MongoDB")
    except Exception as e:
        print(f"Failed to save summary to MongoDB: {str(e)}")

def process_pdf_files(dataset_json):
    """
    Process each PDF file in the dataset and save summaries in MongoDB.
    """
    with open(dataset_json, 'r') as f:
        data = json.load(f)

    combined_summary = []

    for pdf_key, url in data.items():
        pdf_file_name = f"{pdf_key}.pdf"
        pdf_path = os.path.join(app.config['PDF_FOLDER'], pdf_file_name)

        if not os.path.exists(pdf_path):
            if not download_pdf(url, pdf_path):
                continue

        extracted_text = extract_text(pdf_path)

        if "Error" in extracted_text:
            print(f"Text extraction failed for {pdf_file_name}. Falling back to MobileNetV3 OCR.")
            extracted_text = mobile_net_ocr(pdf_path)

        if "Error" not in extracted_text:
            summary = summarize_text(extracted_text)
            keywords = extract_keywords(extracted_text)

            combined_summary.append({
                "original_file": pdf_file_name,
                "summary": summary,
                "keywords": keywords
            })
        else:
            print(f"Failed to process {pdf_file_name}")

    save_summary_to_mongodb(combined_summary)

# Flask Routes

@app.route('/')
def index():
    """
    Render the main page with a list of all PDFs.
    """
        pdfs = list(collection.find({}, {'original_file': 1}))
        app.logger.info(f"Retrieved {len(pdfs)} PDFs from the database")
        return render_template('index.html', pdfs=pdfs)

@app.route('/summary/<pdf_name>')
def get_summary(pdf_name):
    """
    Retrieve and return the summary for a specific PDF.
    """
    try:
        pdf_data = collection.find_one({'original_file': pdf_name})
        if pdf_data:
            app.logger.info(f"Retrieved summary for {pdf_name}")
            return jsonify({
                'summary': pdf_data.get('summary', 'Summary not available'),
                'keywords': pdf_data.get('keywords', [])
            })
        else:
            app.logger.warning(f"PDF not found: {pdf_name}")
            return jsonify({'error': 'PDF not found'}), 404
    except Exception as e:
        app.logger.error(f"Error fetching summary for {pdf_name}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """
    Handle file uploads, process the uploaded PDF, and save its summary.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['PDF_FOLDER'], filename)
            file.save(filepath)
            
            extracted_text = extract_text(filepath)
            if "Error" in extracted_text:
                extracted_text = mobile_net_ocr(filepath)
            
            summary = summarize_text(extracted_text)
            keywords = extract_keywords(extracted_text)
            
            summary_data = {
                "original_file": filename,
                "summary": summary,
                "keywords": keywords
            }
            collection.insert_one(summary_data)
            
            return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/process_dataset')
def process_dataset():
    """
    Process all PDFs in the dataset and save their summaries.
    """
    dataset_json_file = "Dataset.json"
    process_pdf_files(dataset_json_file)
    return redirect(url_for('index'))

@app.route('/pdf/<pdf_name>')
def serve_pdf(pdf_name):
    """
    Serve a specific PDF file.
    """
    pdf_path = os.path.join(app.config['PDF_FOLDER'], pdf_name)
    if os.path.exists(pdf_path):
        return send_file(pdf_path, mimetype='application/pdf')
    else:
        return "PDF not found", 404

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
