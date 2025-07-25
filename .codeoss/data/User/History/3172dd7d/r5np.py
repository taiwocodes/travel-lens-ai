# app.py
# Import necessary libraries
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import json
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, Image
from PIL import Image as PIL_Image # Use an alias to avoid name conflicts
from io import BytesIO
import os

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# --- Configuration ---
# FIX: Increase the maximum file size limit to 16 MB.
# The 413 "Payload Too Large" error happens because the default limit is too small for some images.
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

# It's better to get these from environment variables in a real app
# For the workshop, you can have participants set them here.
# Make sure to remind them NOT to commit these to public Git repositories.
PROJECT_ID = "travel-lens-ai"
LOCATION = "us-central1"      # Or your preferred location


# --- Core AI Analysis Logic from your Colab Notebook ---
def analyze_image_from_bytes(image_bytes):
    """
    Initializes the model and analyzes image bytes using the Gemini Pro Vision model.
    This is the main function adapted from your Colab notebook.
    """
    try:
        # Initialize Vertex AI and the model inside the function.
        # This ensures that any initialization errors (e.g., bad project ID)
        # are caught by this try...except block and returned as a proper JSON error,
        # preventing the entire application from crashing on start.
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        multimodal_model = GenerativeModel("gemini-1.0-pro-vision")

        # Load the image from the byte stream provided by the web request
        image = Image.from_bytes(image_bytes)

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
        response = multimodal_model.generate_content([prompt, image])

        # The model should return a JSON string. We need to parse it.
        # Sometimes the model returns the JSON wrapped in ```json ... ```, so we clean it.
        response_text = response.text.strip().replace("```json", "").replace("```", "")
        
        # Convert the cleaned string to a Python dictionary
        analysis_result = json.loads(response_text)
        
        print("Successfully analyzed image.")
        return analysis_result

    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred during AI analysis: {e}")
        # Return a structured error message. This now catches initialization errors too.
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
