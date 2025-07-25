# app.py
# Import necessary libraries
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import json
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image # Updated import
from PIL import Image as PIL_Image # Use an alias to avoid name conflicts
from io import BytesIO
import os
import google.auth # FIX: Import the google.auth library

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# --- Configuration ---
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

PROJECT_ID = "travel-lens-ai"
LOCATION = "us-central1"
MODEL_NAME = "gemini-1.5-flash-001"


# --- Core AI Analysis Logic from your Colab Notebook ---
def analyze_image_from_bytes(image_bytes):
    """
    Initializes the model and analyzes image bytes using the Gemini model.
    This is the main function adapted from your Colab notebook.
    """
    try:
        # FIX: Add explicit authentication for the Cloud Shell environment.
        # This gets the default credentials from the gcloud environment.
        credentials, project_id = google.auth.default()
        
        # Initialize Vertex AI inside the function, passing the credentials.
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
        
        multimodal_model = GenerativeModel(MODEL_NAME)

        # Load the image from the byte stream provided by the web request
        image_part = Part.from_data(image_bytes, mime_type="image/jpeg")


        # The prompt for the model
        prompt = """
        You are a travel expert. Analyze the image provided and provide the following information in a JSON format:
        - "landmarkName": The name of the landmark, city, or place.
        - "description": A captivating and interesting description of the place.
        - "location": A JSON object with "city" and "country".
        - "personalizedRecommendations": An array of 3-4 personalized and actionable travel tips or recommendations for someone visiting this place.
        - "photoTips": A creative tip for taking a great photo at this location.
        """

        # Send the image and prompt to the model
        response = multimodal_model.generate_content([prompt, image_part])

        # The model should return a JSON string. We need to parse it.
        response_text = response.text.strip().replace("```json", "").replace("```", "")
        
        # Convert the cleaned string to a Python dictionary
        analysis_result = json.loads(response_text)
        
        print("Successfully analyzed image.")
        return analysis_result

    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred during AI analysis: {e}")
        # Return a structured error message.
        return {"error": f"An internal AI error occurred. Please check the backend logs. Details: {e}"}


# --- Flask API Endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze_image_endpoint():
    """
    The main API endpoint. It receives an image, passes it to the
    analysis pipeline, and returns the result.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "No image file selected"}), 400

    try:
        # Read the image file's data in bytes
        image_data = image_file.read()

        # Run your AI analysis pipeline
        analysis_result = analyze_image_from_bytes(image_data)

        # Check if the analysis returned an error
        if "error" in analysis_result:
            return jsonify(analysis_result), 500
            
        # Return the successful analysis as JSON
        return jsonify(analysis_result), 200

    except Exception as e:
        print(f"An error occurred in the endpoint: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

# --- Frontend Route ---
# This route will serve your index.html file
@app.route('/')
def home():
    """Serves the frontend HTML page."""
    return render_template('index.html')

# To run this app:
# 1. Make sure you've run `pip install -r requirements.txt`
# 2. In the Cloud Shell terminal, run: `python app.py`
if __name__ == '__main__':
    # Flask runs on port 5000 by default. Cloud Shell's Web Preview uses port 8080.
    # We'll run on a port that Cloud Shell can easily preview.
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)
