@echo off
setlocal

cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py -m streamlit run app.py
    exit /b %errorlevel%
)

where python >nul 2>nul
if %errorlevel%==0 (
    python -m streamlit run app.py
    exit /b %errorlevel%
)

echo Python was not found on PATH. Install Python, then run:
echo python -m pip install -r requirements.txt
echo python -m streamlit run app.py
exit /b 1
