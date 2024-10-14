from flask import Flask, render_template, jsonify
from pymongo import MongoClient
import os
import logging
from pymongo.server_api import ServerApi
import requests
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# MongoDB setup
uri = "mongodb+srv://Anant31:twMXokWBWL6IQz7h@cluster0.munjj8l.mongodb.net/pdf_summaries?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi('1'))

# MongoDB Database and Collection
db = client['pdf_summaries']  # Create or use the database
collection = db['summaries']   # Create or use the collection

@app.route('/')
def index():
    try:
        # Fetch all PDF names from MongoDB
        pdfs = list(collection.find({}, {'original_file': 1}))
        app.logger.info(f"Retrieved {len(pdfs)} PDFs from the database")
        return render_template('index.html', pdfs=pdfs)
    except Exception as e:
        app.logger.error(f"Error fetching PDFs: {str(e)}")
        return render_template('error.html', error="Failed to fetch PDFs"), 500

@app.route('/summary/<pdf_name>')
def get_summary(pdf_name):
    try:
        # Fetch summary for the specified PDF
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

if __name__ == '__main__':
    app.run(debug=True)# Import necessary libraries
