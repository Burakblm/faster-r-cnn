#!/bin/bash

echo "Creating Python virtual environment..."
python3 -m venv venv

echo "upgrade pip"
pip install --upgrade pip

echo "Activating the virtual environment..."
source venv/bin/activate

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Running main.py..."
python main.py
