#!/bin/bash

# Step 1: Install dependencies
echo "Installing required Python packages..."
pip install -r requirements.txt

# Step 2: Run the Flask application
echo "Starting Flask app..."
python app.py
