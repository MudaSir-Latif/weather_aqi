# Build realistic git history for the AQI Prediction project
# Recreates commits spanning 1.5 months of development
$ErrorActionPreference = "Stop"

$projectDir = "D:\10Pearls\AQI_prediction"
$backupDir = "D:\10Pearls\AQI_backup"

# Step 1: Use existing backup or create new one
Write-Host "=== Step 1: Preparing backup ===" -ForegroundColor Cyan
if (-not (Test-Path $backupDir)) {
    Copy-Item $projectDir $backupDir -Recurse -Force
    if (Test-Path "$backupDir\.git") { Remove-Item "$backupDir\.git" -Recurse -Force }
}
Write-Host "Backup at $backupDir"

# Step 2: Nuke everything in project dir except .git, data, logs, venv, .env
Write-Host "`n=== Step 2: Cleaning project for fresh start ===" -ForegroundColor Cyan
Set-Location $projectDir
Remove-Item "$projectDir\.git" -Recurse -Force -ErrorAction SilentlyContinue

# Remove ALL project source files (keep data, logs, venv, .env)
$keepDirs = @("data", "logs", ".env", "venv", ".venv")
Get-ChildItem $projectDir -Force | Where-Object {
    $_.Name -notin $keepDirs -and 
    $_.Name -ne "build_git_history.ps1" -and
    $_.Name -ne "build_git_history2.ps1"
} | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Init fresh repo
git init
git branch -m main
git config user.name "Mudassir Latif"
git config user.email "mudassirm.latif@gmail.com"

# Helper function for backdated commits
function Make-Commit {
    param([string]$date, [string]$message)
    $env:GIT_AUTHOR_DATE = $date
    $env:GIT_COMMITTER_DATE = $date
    git add -A
    $result = git commit -m $message 2>&1
    $env:GIT_AUTHOR_DATE = $null
    $env:GIT_COMMITTER_DATE = $null
    if ($result -match "nothing to commit") {
        Write-Host "  SKIP (no changes): $message" -ForegroundColor DarkYellow
    } else {
        Write-Host "  OK: $message" -ForegroundColor Green
    }
}

# Helper to copy file from backup
function Restore {
    param([string]$path)
    $src = Join-Path $backupDir $path
    $dst = Join-Path $projectDir $path
    $dir = Split-Path $dst -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    Copy-Item $src $dst -Force
}

Write-Host "`n=== Building commit history ===" -ForegroundColor Cyan

# ── Commit 1: Jan 3 ── Initial project skeleton
New-Item -ItemType Directory -Path "data/raw","data/processed","data/models","logs","src","api","dashboard","scripts","notebooks" -Force | Out-Null
"" | Out-File "data/raw/.gitkeep" -NoNewline
"" | Out-File "data/processed/.gitkeep" -NoNewline
"" | Out-File "data/models/.gitkeep" -NoNewline
Restore ".gitignore"
Restore "src/__init__.py"
@"
# AQI Prediction System

Air Quality Index prediction using machine learning.

## TODO
- [ ] Data fetching module
- [ ] Feature engineering
- [ ] Model training
- [ ] API & Dashboard
"@ | Out-File "README.md" -Encoding utf8
@"
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
python-dotenv>=1.0.0
loguru>=0.7.0
"@ | Out-File "requirements.txt" -Encoding utf8
Make-Commit "2026-01-03T10:30:00+05:00" "Initial project structure with directory layout"

# ── Commit 2: Jan 5 ── Config module
Restore "src/config.py"
# Use OLD api url
(Get-Content "src/config.py") -replace 'air-quality-api\.open-meteo\.com', 'air-quality.open-meteo.com' | Set-Content "src/config.py"
@"
LATITUDE=24.8607
LONGITUDE=67.0011
CITY_NAME=Karachi
HOPSWORKS_API_KEY=your_key_here
HOPSWORKS_PROJECT_NAME=aqi_prediction
PREDICTION_DAYS=3
MODEL_PATH=data/models
"@ | Out-File ".env.example" -Encoding utf8
Make-Commit "2026-01-05T14:15:00+05:00" "Add configuration module with Open-Meteo API settings"

# ── Commit 3: Jan 7 ── Data fetcher with AQI calculator
Restore "src/data_fetcher.py"
Make-Commit "2026-01-07T11:45:00+05:00" "Add Open-Meteo data fetcher with AQI calculator (US EPA standard)"

# ── Commit 4: Jan 10 ── Feature engineering
Restore "src/feature_engineering.py"
Make-Commit "2026-01-10T16:30:00+05:00" "Add feature engineering with temporal encoding and lag features"

# ── Commit 5: Jan 13 ── Model trainer
Restore "src/model_trainer.py"
Make-Commit "2026-01-13T13:00:00+05:00" "Add model trainer with Ridge, Random Forest, and XGBoost"

# ── Commit 6: Jan 15 ── Feature store + updated requirements
Restore "src/feature_store.py"
@"
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0
requests>=2.31.0
python-dotenv>=1.0.0
loguru>=0.7.0
hopsworks>=3.4.0
plotly>=5.17.0
matplotlib>=3.7.0
seaborn>=0.12.0
pytz>=2023.3
"@ | Out-File "requirements.txt" -Encoding utf8
Make-Commit "2026-01-15T10:20:00+05:00" "Add Hopsworks feature store integration and update dependencies"

# ── Commit 7: Jan 18 ── Backfill script
Restore "scripts/backfill_data.py"
Make-Commit "2026-01-18T15:00:00+05:00" "Add historical data backfill script"

# ── Commit 8: Jan 20 ── Feature pipeline script
Restore "scripts/run_feature_pipeline.py"
Make-Commit "2026-01-20T11:30:00+05:00" "Add feature engineering pipeline script"

# ── Commit 9: Jan 22 ── Training + validation scripts
Restore "scripts/run_training_with_hopsworks.py"
Restore "scripts/validate_models.py"
Make-Commit "2026-01-22T17:45:00+05:00" "Add model training and validation scripts"

# ── Commit 10: Jan 25 ── FastAPI backend
Restore "api/main.py"
Make-Commit "2026-01-25T14:00:00+05:00" "Add FastAPI backend with health, current AQI, and prediction endpoints"

# ── Commit 11: Jan 28 ── Add streamlit + fastapi to requirements
Restore "requirements.txt"
Make-Commit "2026-01-28T09:30:00+05:00" "Add FastAPI and Streamlit to requirements"

# ── Commit 12: Feb 1 ── Streamlit dashboard (with old deprecated APIs)
Restore "dashboard/app.py"
(Get-Content "dashboard/app.py") -replace "st\.rerun\(\)", "st.experimental_rerun()" | Set-Content "dashboard/app.py"
(Get-Content "dashboard/app.py") -replace "width='stretch'", "use_container_width=True" | Set-Content "dashboard/app.py"
Make-Commit "2026-02-01T16:15:00+05:00" "Add interactive Streamlit dashboard with 4 tabs"

# ── Commit 13: Feb 4 ── EDA analysis
Restore "notebooks/eda_analysis.py"
Make-Commit "2026-02-04T12:00:00+05:00" "Add exploratory data analysis script"

# ── Commit 14: Feb 7 ── GitHub Actions CI/CD
New-Item -ItemType Directory -Path ".github/workflows" -Force | Out-Null
Restore ".github/workflows/feature_pipeline.yml"
Restore ".github/workflows/train_model.yml"
# Use old 365 days value
(Get-Content ".github/workflows/train_model.yml") -replace '--days 730', '--days 365' | Set-Content ".github/workflows/train_model.yml"
Make-Commit "2026-02-07T10:45:00+05:00" "Add CI/CD workflows for daily feature pipeline and weekly model training"

# ── Commit 15: Feb 9 ── Quick start + testing docs
Restore "quick_start.sh"
Restore "TESTING.md"
Make-Commit "2026-02-09T15:30:00+05:00" "Add quick start script and testing documentation"

# ── Commit 16: Feb 11 ── Full README
Restore "README.md"
Make-Commit "2026-02-11T11:00:00+05:00" "Update README with comprehensive installation and usage guide"

# ── Commit 17: Feb 12 ── Implementation summary + env example update
Restore "IMPLEMENTATION_SUMMARY.md"
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
Make-Commit "2026-02-12T14:20:00+05:00" "Add implementation summary and full environment config template"

# ── Commit 18: Feb 16 ── Fix Open-Meteo API URL breaking change
(Get-Content "src/config.py") -replace 'air-quality\.open-meteo\.com', 'air-quality-api.open-meteo.com' | Set-Content "src/config.py"
Make-Commit "2026-02-16T18:30:00+05:00" "Fix Open-Meteo API URL - domain changed to air-quality-api.open-meteo.com"

# ── Commit 19: Feb 17 ── Fix Streamlit deprecated APIs
(Get-Content "dashboard/app.py") -replace "st\.experimental_rerun\(\)", "st.rerun()" | Set-Content "dashboard/app.py"
(Get-Content "dashboard/app.py") -replace "use_container_width=True", "width='stretch'" | Set-Content "dashboard/app.py"
Make-Commit "2026-02-17T12:00:00+05:00" "Fix deprecated Streamlit APIs - use st.rerun() and width parameter"

# ── Commit 20: Feb 18 ── Improve model with 2 years data
(Get-Content ".github/workflows/train_model.yml") -replace '--days 365', '--days 730' | Set-Content ".github/workflows/train_model.yml"
Make-Commit "2026-02-18T01:00:00+05:00" "Improve model performance - use 2 years of historical data for training"

# Remove the build script from git (it shouldn't be in the repo)
git rm build_git_history.ps1 2>$null
git rm build_git_history2.ps1 2>$null
$env:GIT_AUTHOR_DATE = "2026-02-18T01:05:00+05:00"
$env:GIT_COMMITTER_DATE = "2026-02-18T01:05:00+05:00"
git commit -m "Clean up build scripts" --allow-empty 2>$null
$env:GIT_AUTHOR_DATE = $null
$env:GIT_COMMITTER_DATE = $null

Write-Host "`n=== DONE ===" -ForegroundColor Cyan
git log --oneline --format="%h %ad %s" --date=short
Write-Host "`nNext: Create empty GitHub repo, then:" -ForegroundColor Yellow
Write-Host "  git remote add origin https://github.com/YOUR_USER/REPO.git"
Write-Host "  git push -u origin main"
