@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo BITS Pilani ML Assignment 2 - Setup and Launch
echo ============================================================

where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python is not available on PATH.
  echo Install Python 3.10 or 3.11, then run this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating a local virtual environment...
  python -m venv .venv
  if errorlevel 1 goto :failed
)

echo Installing exact project dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo Verifying the five bundled model files...
if not exist "models\logistic_regression.joblib" goto :train
if not exist "models\decision_tree.joblib" goto :train
if not exist "models\knn.joblib" goto :train
if not exist "models\naive_bayes.joblib" goto :train
if not exist "models\svm.joblib" goto :train
goto :launch

:train
echo One or more model files are missing. Reproducing all models...
".venv\Scripts\python.exe" train_models.py
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" generate_report.py
if errorlevel 1 goto :failed

:launch
echo Launching Streamlit. Keep this window open.
".venv\Scripts\python.exe" -m streamlit run app.py
goto :end

:failed
echo.
echo SETUP FAILED. Read the error above, check internet access, and retry.
pause
exit /b 1

:end
endlocal
