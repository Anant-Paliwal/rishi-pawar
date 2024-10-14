PDF Summarizer with MongoDB and Flask
This project is a Flask-based web application that processes and summarizes PDF documents. It utilizes PyPDF2 for text extraction, MobileNetV3 for image-based text recognition (OCR), and stores the processed summaries and keywords in a MongoDB database. Additionally, the app allows users to upload PDFs, extract their summaries, and view previously processed documents.

Features
PDF Processing: Extract text from PDF files using PyPDF2.
OCR Backup: If text extraction fails, fallback to MobileNetV3-based OCR for text recognition.
Text Summarization: Automatically generate concise summaries from extracted text using word frequency analysis.
Keyword Extraction: Extract key terms from PDF text.
MongoDB Storage: Summaries and keywords are saved to a MongoDB collection.
File Upload: Allows users to upload their own PDF files for processing.
File Serving: Users can download or view processed PDFs directly from the app.
Tech Stack
Flask: Python web framework to handle routes and logic.
MongoDB: NoSQL database to store summaries and keywords.
PyPDF2: Extract text from PDFs.
MobileNetV3: Image classification model used for OCR fallback.
NLTK: Natural Language Toolkit for text processing and summarization.
TensorFlow: Used for MobileNetV3 inference.
Pillow: For image processing.
Prerequisites
To run this project, you need the following installed:

Python 3.x
MongoDB
Pipenv (or use pip)
Setup
1. Clone the repository
bash
Copy code
git clone https://github.com/yourusername/pdf-summarizer-flask.git
cd pdf-summarizer-flask
2. Create a virtual environment and install dependencies
bash
Copy code
pipenv install
pipenv shell
Alternatively, if you're using pip:

bash
Copy code
pip install -r requirements.txt
3. Set up MongoDB
Ensure that you have a MongoDB instance running and update the uri in the code to reflect your connection string:

python
Copy code
uri = "your-mongodb-uri"
4. Run the Flask application
bash
Copy code
flask run
5. Access the application
Open your browser and navigate to:

arduino
Copy code
http://127.0.0.1:5000/
Routes
/ - Main page displaying all processed PDFs.
/upload - Upload new PDFs for processing.
/summary/<pdf_name> - View summary and keywords for a specific PDF.
/process_dataset - Process all PDFs from a dataset stored in a JSON file.
/pdf/<pdf_name> - Download or view a specific PDF.
File Upload
To upload a PDF:

Go to the /upload route.
Choose a valid PDF file.
Submit the form to upload and process the PDF.
Dataset Processing
If you want to process a batch of PDFs stored in a dataset, you can place the dataset URLs in a Dataset.json file and then trigger processing by navigating to /process_dataset.

Sample Dataset.json
json
Copy code
{
  "pdf_1": "http://example.com/sample1.pdf",
  "pdf_2": "http://example.com/sample2.pdf"
}
Unit Testing
Unit tests for the application are located in the tests/ directory. You can run the tests using pytest.

bash
Copy code
pytest
The tests cover the following:

PDF text extraction
Text summarization
Keyword extraction
File uploads
MongoDB storage
Logging
The app uses Python's logging module to log key events, errors, and user actions. Logs are displayed in the terminal when the app is running.

Future Enhancements
Improve OCR accuracy by incorporating more advanced deep learning models.
Add user authentication for secure file uploads and viewing.
Provide detailed analytics on the summaries and keywords.
Implement a REST API for external services to access the summarization features.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Contributing
Feel free to open issues or submit pull requests if you have suggestions or improvements for this project.
