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