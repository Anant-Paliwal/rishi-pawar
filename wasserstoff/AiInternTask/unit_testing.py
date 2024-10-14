import unittest
import os
from app import app, allowed_file, extract_text, summarize_text, extract_keywords
from flask import url_for

class FlaskAppTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up the Flask test client and initialize any test data.
        """
        cls.client = app.test_client()
        app.config['TESTING'] = True

        # Create necessary directories for testing
        if not os.path.exists(app.config['PDF_FOLDER']):
            os.makedirs(app.config['PDF_FOLDER'])

    @classmethod
    def tearDownClass(cls):
        """
        Clean up any resources used during the tests.
        """
        for file in os.listdir(app.config['PDF_FOLDER']):
            os.remove(os.path.join(app.config['PDF_FOLDER'], file))

    def test_home_page(self):
        """
        Test if the home page loads correctly.
        """
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Upload a PDF', response.data)

    def test_upload_valid_pdf(self):
        """
        Test uploading a valid PDF file.
        """
        # Simulate uploading a file
        pdf_data = {
            'file': (open('sample.pdf', 'rb'), 'sample.pdf')
        }
        response = self.client.post('/upload', data=pdf_data, content_type='multipart/form-data')

        # Verify redirection after upload
        self.assertEqual(response.status_code, 302)
        self.assertIn('/pdf', response.headers['Location'])

    def test_invalid_file_upload(self):
        """
        Test uploading an invalid file (not a PDF).
        """
        invalid_data = {
            'file': (open('sample.txt', 'rb'), 'sample.txt')
        }
        response = self.client.post('/upload', data=invalid_data, content_type='multipart/form-data')

        # Verify that it redirects back (no upload allowed)
        self.assertEqual(response.status_code, 302)

    def test_allowed_file(self):
        """
        Test the file type validation function.
        """
        self.assertTrue(allowed_file('document.pdf'))
        self.assertFalse(allowed_file('document.txt'))

    def test_extract_text(self):
        """
        Test the text extraction from a PDF.
        """
        sample_pdf = 'sample.pdf'
        extracted_text = extract_text(sample_pdf)
        self.assertIsInstance(extracted_text, str)
        self.assertNotEqual(extracted_text, '')

    def test_summarize_text(self):
        """
        Test the summary generation from text.
        """
        text = "This is a sample text used to generate a summary. The summary should be concise."
        summary = summarize_text(text)
        self.assertIsInstance(summary, str)
        self.assertIn("summary", summary.lower())

    def test_extract_keywords(self):
        """
        Test the keyword extraction from text.
        """
        text = "This is a sample text used to extract keywords from."
        keywords = extract_keywords(text)
        self.assertIsInstance(keywords, list)
        self.assertGreater(len(keywords), 0)

    def test_get_summary(self):
        """
        Test fetching a summary for a specific PDF.
        """
        response = self.client.get('/summary/sample.pdf')
        self.assertEqual(response.status_code, 200)
        self.assertIn('summary', response.get_json())

    def test_serve_pdf(self):
        """
        Test serving a PDF file.
        """
        sample_pdf = 'sample.pdf'
        # Add a dummy file to the folder for testing
        with open(os.path.join(app.config['PDF_FOLDER'], sample_pdf), 'wb') as f:
            f.write(b"Dummy content")

        response = self.client.get(f'/pdf/{sample_pdf}')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'application/pdf')

if __name__ == '__main__':
    unittest.main()
