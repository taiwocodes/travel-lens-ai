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