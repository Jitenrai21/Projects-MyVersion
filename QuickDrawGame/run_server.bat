@echo off
echo Starting QuickDraw 21-Class API Server...
echo.
echo Installing requirements if needed...
pip install -r requirements.txt
echo.
echo Starting server on http://localhost:8000
echo Game will be available at http://localhost:8000/static/index.html
echo API documentation at http://localhost:8000/docs
echo.
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
