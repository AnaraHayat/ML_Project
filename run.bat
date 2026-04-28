@echo off
echo Starting MindGuard AI...
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Starting FastAPI server on http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

cd src
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
