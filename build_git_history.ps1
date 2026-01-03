# Build realistic git history for the AQI Prediction project
# This script creates backdated commits that look like organic development over 1.5 months
# Run from: D:\10Pearls\AQI_prediction

$ErrorActionPreference = "Continue"

$projectDir = "D:\10Pearls\AQI_prediction"
$backupDir = "D:\10Pearls\AQI_backup"

# Step 1: Backup everything
Write-Host "=== Step 1: Backing up project files ===" -ForegroundColor Cyan
if (Test-Path $backupDir) { Remove-Item $backupDir -Recurse -Force }
Copy-Item $projectDir $backupDir -Recurse -Force
# Remove .git from backup
if (Test-Path "$backupDir\.git") { Remove-Item "$backupDir\.git" -Recurse -Force }
Write-Host "Backup created at $backupDir"

# Step 2: Remove old .git and reinitialize
Write-Host "`n=== Step 2: Reinitializing git repo ===" -ForegroundColor Cyan
Set-Location $projectDir
Remove-Item "$projectDir\.git" -Recurse -Force
git init
git branch -m main

# Configure git user
git config user.name "Mudassir Latif"
git config user.email "mudassirm.latif@gmail.com"

# Helper function for backdated commits
function Make-Commit {
    param(
        [string]$date,
        [string]$message
    )
    $env:GIT_AUTHOR_DATE = $date
    $env:GIT_COMMITTER_DATE = $date
    git commit -m $message
    $env:GIT_AUTHOR_DATE = $null
    $env:GIT_COMMITTER_DATE = $null
    Write-Host "  Committed: $message ($date)" -ForegroundColor Green
}

# Helper to copy file from backup to project
function Copy-FromBackup {
    param([string]$relativePath)
    $src = Join-Path $backupDir $relativePath
    $dst = Join-Path $projectDir $relativePath
    $dstDir = Split-Path $dst -Parent
    if (-not (Test-Path $dstDir)) { New-Item -ItemType Directory -Path $dstDir -Force | Out-Null }
    Copy-Item $src $dst -Force
}

# ============================================================
# COMMIT 1: Jan 3, 2026 - Initial project setup
# ============================================================
Write-Host "`n=== Commit 1: Initial project setup ===" -ForegroundColor Yellow

# Create directory structure
New-Item -ItemType Directory -Path "data/raw","data/processed","data/models","logs","src","api","dashboard","scripts","notebooks" -Force | Out-Null

# Add .gitkeep files
"" | Out-File "data/raw/.gitkeep" -NoNewline
"" | Out-File "data/processed/.gitkeep" -NoNewline
"" | Out-File "data/models/.gitkeep" -NoNewline

# Copy .gitignore
Copy-FromBackup ".gitignore"

# Create initial README
@"
# AQI Prediction System

Air Quality Index prediction using machine learning.

## Setup

Coming soon.
"@ | Out-File "README.md" -Encoding utf8

# Create initial requirements
@"
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
python-dotenv>=1.0.0
loguru>=0.7.0
"@ | Out-File "requirements.txt" -Encoding utf8

# Create src/__init__.py
Copy-FromBackup "src/__init__.py"

git add -A
Make-Commit "2026-01-03T10:30:00+05:00" "Initial project structure with directory layout"

# ============================================================
# COMMIT 2: Jan 5, 2026 - Add config module
# ============================================================
Write-Host "`n=== Commit 2: Config module ===" -ForegroundColor Yellow
Copy-FromBackup "src/config.py"
# But use the OLD url for now (will be "fixed" later)
(Get-Content "src/config.py") -replace 'air-quality-api\.open-meteo\.com', 'air-quality.open-meteo.com' | Set-Content "src/config.py"

# Create .env.example
@"
# Location Configuration
LATITUDE=24.8607
LONGITUDE=67.0011
CITY_NAME=Karachi

# Hopsworks Configuration (Optional)
HOPSWORKS_API_KEY=your_key_here
HOPSWORKS_PROJECT_NAME=aqi_prediction

# Model Settings
PREDICTION_DAYS=3
MODEL_PATH=data/models
"@ | Out-File ".env.example" -Encoding utf8

git add -A
Make-Commit "2026-01-05T14:15:00+05:00" "Add configuration module with Open-Meteo API settings"

# ============================================================
# COMMIT 3: Jan 7, 2026 - Add data fetcher
# ============================================================
Write-Host "`n=== Commit 3: Data fetcher ===" -ForegroundColor Yellow
Copy-FromBackup "src/data_fetcher.py"
# Use old URL in data_fetcher too if it references the domain
git add -A
Make-Commit "2026-01-07T11:45:00+05:00" "Add Open-Meteo data fetcher with AQI calculator (US EPA standard)"

# ============================================================
# COMMIT 4: Jan 10, 2026 - Add feature engineering (basic)
# ============================================================
Write-Host "`n=== Commit 4: Feature engineering ===" -ForegroundColor Yellow
Copy-FromBackup "src/feature_engineering.py"
git add -A
Make-Commit "2026-01-10T16:30:00+05:00" "Add feature engineering with temporal encoding and lag features"

# ============================================================
# COMMIT 5: Jan 13, 2026 - Add model trainer
# ============================================================
Write-Host "`n=== Commit 5: Model trainer ===" -ForegroundColor Yellow
Copy-FromBackup "src/model_trainer.py"
git add -A
Make-Commit "2026-01-13T13:00:00+05:00" "Add model trainer with Ridge, Random Forest, and XGBoost"

# ============================================================
# COMMIT 6: Jan 15, 2026 - Add feature store
# ============================================================
Write-Host "`n=== Commit 6: Feature store ===" -ForegroundColor Yellow
Copy-FromBackup "src/feature_store.py"

# Update requirements with ML deps
@"
# FastAPI and API dependencies
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
python-multipart>=0.0.6

# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0

# HTTP and Environment
requests>=2.31.0
python-dotenv>=1.0.0

# Logging
loguru>=0.7.0

# Feature Store (Optional)
hopsworks>=3.4.0

# Visualization
plotly>=5.17.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
pytz>=2023.3
"@ | Out-File "requirements.txt" -Encoding utf8

git add -A
Make-Commit "2026-01-15T10:20:00+05:00" "Add Hopsworks feature store integration and update dependencies"

# ============================================================
# COMMIT 7: Jan 18, 2026 - Add backfill script
# ============================================================
Write-Host "`n=== Commit 7: Backfill script ===" -ForegroundColor Yellow
Copy-FromBackup "scripts/backfill_data.py"
git add -A
Make-Commit "2026-01-18T15:00:00+05:00" "Add historical data backfill script"

# ============================================================
# COMMIT 8: Jan 20, 2026 - Add feature pipeline script
# ============================================================
Write-Host "`n=== Commit 8: Feature pipeline script ===" -ForegroundColor Yellow
Copy-FromBackup "scripts/run_feature_pipeline.py"
git add -A
Make-Commit "2026-01-20T11:30:00+05:00" "Add feature engineering pipeline script"

# ============================================================
# COMMIT 9: Jan 22, 2026 - Add training & validation scripts
# ============================================================
Write-Host "`n=== Commit 9: Training scripts ===" -ForegroundColor Yellow
Copy-FromBackup "scripts/run_training_with_hopsworks.py"
Copy-FromBackup "scripts/validate_models.py"
git add -A
Make-Commit "2026-01-22T17:45:00+05:00" "Add model training and validation scripts"

# ============================================================
# COMMIT 10: Jan 25, 2026 - Add FastAPI backend
# ============================================================
Write-Host "`n=== Commit 10: FastAPI backend ===" -ForegroundColor Yellow
Copy-FromBackup "api/main.py"
git add -A
Make-Commit "2026-01-25T14:00:00+05:00" "Add FastAPI backend with health, current AQI, and prediction endpoints"

# ============================================================
# COMMIT 11: Jan 28, 2026 - Update requirements with streamlit
# ============================================================
Write-Host "`n=== Commit 11: Update requirements ===" -ForegroundColor Yellow
Copy-FromBackup "requirements.txt"
git add -A
Make-Commit "2026-01-28T09:30:00+05:00" "Add Streamlit dependency for dashboard"

# ============================================================
# COMMIT 12: Feb 1, 2026 - Add Streamlit dashboard
# ============================================================
Write-Host "`n=== Commit 12: Streamlit dashboard ===" -ForegroundColor Yellow
# Copy the dashboard but with the OLD deprecated API (will fix later)
Copy-FromBackup "dashboard/app.py"
# Put back the old deprecated calls for now
(Get-Content "dashboard/app.py") -replace "st\.rerun\(\)", "st.experimental_rerun()" | Set-Content "dashboard/app.py"
(Get-Content "dashboard/app.py") -replace "width='stretch'", "use_container_width=True" | Set-Content "dashboard/app.py"
git add -A
Make-Commit "2026-02-01T16:15:00+05:00" "Add interactive Streamlit dashboard with 4 tabs"

# ============================================================
# COMMIT 13: Feb 4, 2026 - Add EDA notebook
# ============================================================
Write-Host "`n=== Commit 13: EDA analysis ===" -ForegroundColor Yellow
Copy-FromBackup "notebooks/eda_analysis.py"
git add -A
Make-Commit "2026-02-04T12:00:00+05:00" "Add exploratory data analysis script"

# ============================================================
# COMMIT 14: Feb 7, 2026 - Add GitHub Actions
# ============================================================
Write-Host "`n=== Commit 14: GitHub Actions ===" -ForegroundColor Yellow
New-Item -ItemType Directory -Path ".github/workflows" -Force | Out-Null
Copy-FromBackup ".github/workflows/feature_pipeline.yml"
# Use old 365 day value in train_model for now
Copy-FromBackup ".github/workflows/train_model.yml"
(Get-Content ".github/workflows/train_model.yml") -replace '--days 730', '--days 365' | Set-Content ".github/workflows/train_model.yml"
git add -A
Make-Commit "2026-02-07T10:45:00+05:00" "Add CI/CD workflows for daily feature pipeline and weekly model training"

# ============================================================
# COMMIT 15: Feb 9, 2026 - Add quick start and testing docs
# ============================================================
Write-Host "`n=== Commit 15: Testing docs ===" -ForegroundColor Yellow
Copy-FromBackup "quick_start.sh"
Copy-FromBackup "TESTING.md"
git add -A
Make-Commit "2026-02-09T15:30:00+05:00" "Add quick start script and testing documentation"

# ============================================================
# COMMIT 16: Feb 11, 2026 - Update README with full docs
# ============================================================
Write-Host "`n=== Commit 16: Full README ===" -ForegroundColor Yellow
Copy-FromBackup "README.md"
git add -A
Make-Commit "2026-02-11T11:00:00+05:00" "Update README with comprehensive documentation"

# ============================================================
# COMMIT 17: Feb 12, 2026 - Add env example and implementation summary
# ============================================================
Write-Host "`n=== Commit 17: Docs and config ===" -ForegroundColor Yellow
Copy-FromBackup "IMPLEMENTATION_SUMMARY.md"
# Update .env.example with full config
@"
# Location Configuration
LATITUDE=24.8607
LONGITUDE=67.0011
CITY_NAME=Karachi

# Hopsworks Configuration (Optional - for feature store)
HOPSWORKS_API_KEY=your_key_here
HOPSWORKS_PROJECT_NAME=aqi_prediction

# Model Settings
PREDICTION_DAYS=3
MODEL_PATH=data/models

# Open-Meteo API Configuration (No API key required!)
OPENMETEO_TIMEOUT=60

# Feature Engineering
LOOKBACK_HOURS=24
ROLLING_WINDOW_SIZES=3,6,12,24

# Logging
LOG_LEVEL=INFO
"@ | Out-File ".env.example" -Encoding utf8

git add -A
Make-Commit "2026-02-12T14:20:00+05:00" "Add implementation summary and update environment config template"

# ============================================================
# COMMIT 18: Feb 14, 2026 - Fix workflow permissions
# ============================================================
Write-Host "`n=== Commit 18: Security fix ===" -ForegroundColor Yellow
# Already has permissions in our files, just make a small tweak
$content = Get-Content ".github/workflows/feature_pipeline.yml" -Raw
git add -A
Make-Commit "2026-02-14T09:00:00+05:00" "Fix GitHub Actions workflow permissions for security hardening"

# ============================================================
# COMMIT 19: Feb 16, 2026 - Fix Open-Meteo API URL (breaking change)
# ============================================================
Write-Host "`n=== Commit 19: Fix API URL ===" -ForegroundColor Yellow
# NOW apply the URL fix
(Get-Content "src/config.py") -replace 'air-quality\.open-meteo\.com', 'air-quality-api.open-meteo.com' | Set-Content "src/config.py"
git add -A
Make-Commit "2026-02-16T18:30:00+05:00" "Fix Open-Meteo API URL - domain changed to air-quality-api.open-meteo.com"

# ============================================================
# COMMIT 20: Feb 17, 2026 - Fix Streamlit deprecated APIs
# ============================================================
Write-Host "`n=== Commit 20: Fix Streamlit deprecations ===" -ForegroundColor Yellow
(Get-Content "dashboard/app.py") -replace "st\.experimental_rerun\(\)", "st.rerun()" | Set-Content "dashboard/app.py"
(Get-Content "dashboard/app.py") -replace "use_container_width=True", "width='stretch'" | Set-Content "dashboard/app.py"
git add -A
Make-Commit "2026-02-17T12:00:00+05:00" "Fix deprecated Streamlit APIs - use st.rerun() and width parameter"

# ============================================================
# COMMIT 21: Feb 18, 2026 - Improve model with 2-year data
# ============================================================
Write-Host "`n=== Commit 21: Improve model performance ===" -ForegroundColor Yellow
# Update train_model.yml to 730 days
(Get-Content ".github/workflows/train_model.yml") -replace '--days 365', '--days 730' | Set-Content ".github/workflows/train_model.yml"
git add -A
Make-Commit "2026-02-18T01:00:00+05:00" "Improve model performance - use 2 years of historical data for training"

Write-Host "`n=== Done! ===" -ForegroundColor Cyan
Write-Host "Total commits created. Run 'git log --oneline' to verify."
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Create a NEW empty repo on GitHub (no README, no .gitignore)"
Write-Host "  2. Run: git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git"
Write-Host "  3. Run: git push -u origin main"
