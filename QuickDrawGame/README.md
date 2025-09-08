# QuickDraw 21-Class Game

An AI-powered drawing recognition game that can classify drawings from 21 different object categories using a TensorFlow model trained on Google's QuickDraw dataset.

## 🎨 Available Classes

The AI can recognize drawings of these 21 objects:
- ✈️ Airplane
- ⏰ Alarm Clock  
- 🍎 Apple
- 🍌 Banana
- 🚲 Bicycle
- 🐦 Bird
- 🚗 Car
- 🐱 Cat
- 🪑 Chair
- 🕐 Clock
- 🐶 Dog
- 🐘 Elephant
- 🐟 Fish
- 🌸 Flower
- 🏠 House
- 🍦 Ice Cream
- ✏️ Pencil
- 🍕 Pizza
- 🕷️ Spider
- 🌳 Tree
- ☂️ Umbrella

## 🚀 Quick Start

### Option 1: Using the batch file (Windows)
```bash
# Simply run the included batch file
run_server.bat
```

### Option 2: Manual setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 🎮 How to Play

1. Open your browser and go to: `http://localhost:8000/static/index.html`
2. Click "Start Game"
3. You'll be given a random object to draw
4. Draw it on the canvas within 30 seconds
5. The AI will analyze your drawing and make a prediction
6. See how well the AI recognized your artwork!

## 🏗️ Project Structure

```
QuickDrawGame/
├── frontend/                 # React-like vanilla JS frontend
│   ├── assets/              # Static assets
│   ├── css/                 # Stylesheets
│   ├── index.html           # Main game interface
│   └── script.js            # Game logic
├── backend/                 # FastAPI backend
│   └── app/
│       ├── models/          # ML model logic
│       ├── routes/          # API endpoints
│       └── main.py          # FastAPI app
├── model_training/          # Model training code
│   ├── models/             # Trained model files
│   └── LoadData.py         # Data preprocessing
├── requirements.txt         # Python dependencies
├── run_server.bat          # Quick start script
└── README.md               # This file
```

## 🔧 API Endpoints

- `GET /` - API information
- `GET /static/index.html` - Play the game
- `GET /docs` - Interactive API documentation
- `POST /api/recognize-drawing` - Submit drawing for recognition
- `GET /api/random-object` - Get random object to draw
- `GET /api/model-info` - Model information
- `GET /api/health` - Health check

## 🤖 Model Details

- **Input**: 28x28 grayscale images
- **Architecture**: CNN with Conv2D, MaxPooling, Dense layers
- **Classes**: 21 QuickDraw categories
- **Framework**: TensorFlow/Keras

## 🎯 Game Features

- Real-time drawing canvas
- 30-second timer
- AI confidence scoring
- Top 3 predictions display
- Responsive design for mobile/desktop
- Keyboard shortcuts (C to clear, Enter to start/restart)

## 🛠️ Technical Features

- FastAPI backend with automatic API docs
- CORS enabled for cross-origin requests
- Static file serving for frontend
- Error handling and logging
- Model prediction with confidence scores
- Preprocessing pipeline for drawings

## 📝 Notes

- The model expects 28x28 pixel images (QuickDraw format)
- Drawings are converted from canvas coordinates to model input automatically
- The AI works best with clear, recognizable drawings
- Try different drawing styles if the AI doesn't recognize your artwork!

## 🎨 Tips for Better Recognition

1. Draw clearly and avoid too much detail
2. Make sure your drawing fills a good portion of the canvas
3. Draw the most recognizable features of the object
4. Keep it simple - the model was trained on simple sketches
5. Don't worry if it doesn't get it right every time - even humans disagree on drawings!

Enjoy playing QuickDraw! 🎨✨
