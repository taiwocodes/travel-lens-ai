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
# It's better to get these from environment variables in a real app
# For the workshop, you can have participants set them here.
# Make sure to remind them NOT to commit these to public Git repositories.
PROJECT_ID = "YOUR_GOOGLE_CLOUD_PROJECT_ID"  # <-- IMPORTANT: REPLACE WITH YOUR PROJECT ID
LOCATION = "us-central1"      # Or your preferred location

# Initialize Vertex AI once when the app starts
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Load the Gemini model
multimodal_model = GenerativeModel("gemini-1.0-pro-vision")

# --- Core AI Analysis Logic from your Colab Notebook ---
def analyze_image_from_bytes(image_bytes):
    """
    Analyzes image bytes using the Gemini Pro Vision model.
    This is the main function adapted from your Colab notebook.
    """
    try:
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
        # Return a structured error message
        return {"error": "Failed to analyze the image with the AI model."}