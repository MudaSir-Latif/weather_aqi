#!/bin/bash

# Quick Start Script for AQI Prediction System
# This script helps you get started with the AQI Prediction System

set -e

echo "🌍 AQI Prediction System - Quick Start"
echo "======================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1)
echo "✅ $python_version"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt -q
echo "✅ Dependencies installed"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created"
    echo "⚠️  Please edit .env to set your location (LATITUDE, LONGITUDE, CITY_NAME)"
    echo ""
else
    echo "✅ .env file already exists"
    echo ""
fi

# Create directories
echo "Creating data directories..."
mkdir -p data/raw data/processed data/models logs
echo "✅ Directories created"
echo ""

echo "======================================="
echo "🎉 Setup Complete!"
echo "======================================="
echo ""
echo "Next Steps:"
echo ""
echo "1. Edit .env to set your location coordinates"
echo ""
echo "2. Fetch historical data (takes a few minutes):"
echo "   python scripts/backfill_data.py --days 90"
echo ""
echo "3. Run feature pipeline:"
echo "   python scripts/run_feature_pipeline.py --input data/raw/historical_data.csv"
echo ""
echo "4. Train models:"
echo "   python scripts/run_training_with_hopsworks.py"
echo ""
echo "5. Start the API server:"
echo "   python api/main.py"
echo "   (Available at http://localhost:8000)"
echo ""
echo "6. OR start the dashboard:"
echo "   streamlit run dashboard/app.py"
echo "   (Available at http://localhost:8501)"
echo ""
echo "For more information, see README.md"
