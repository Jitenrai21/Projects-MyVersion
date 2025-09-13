// Get references to the HTML elements
const startScreen = document.getElementById("start-screen");
const gameScreen = document.getElementById("game-screen");
const postGameScreen = document.getElementById("post-game-screen");
const startButton = document.getElementById("start-button");
const restartButton = document.getElementById("restart-button");
const timeLeftDisplay = document.getElementById("time-left");
const currentObjectDisplay = document.getElementById("current-object");
const objectPlaceholder = document.getElementById("object-placeholder");
const clearButton = document.getElementById("clear-button");
const modelGuessDisplay = document.getElementById("model-guess");
const canvas = document.getElementById("drawing-canvas");
const ctx = canvas.getContext("2d");

// Game state variables
let drawing = false;
let lastX = 0;
let lastY = 0;
let timeLeft = 30;
let timer;
let currentObject = "";

// Array to hold the sequence of drawing coordinates
let drawingData = [];

// Set canvas size
canvas.width = 600;
canvas.height = 400;

// Configure canvas for better drawing
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = '#000';
ctx.lineWidth = 5;

// API base URL - adjust if your backend runs on different port
const API_BASE_URL = window.location.origin.includes('localhost') ? 
    'http://localhost:8000' : window.location.origin;

// Initialize the game
document.addEventListener('DOMContentLoaded', function() {
    initializeGame();
});

async function initializeGame() {
    try {
        // Check if the model is loaded
        const modelInfo = await fetch(`${API_BASE_URL}/api/model-info`);
        const info = await modelInfo.json();
        
        if (info.error) {
            console.error('Model not loaded:', info.error);
            alert('⚠️ Model not loaded. Please check the backend.');
            return;
        }
        
        console.log('✅ Model loaded successfully:', info);
        
        // Get initial random object
        await getNewObject();
        
    } catch (error) {
        console.error('Error initializing game:', error);
        alert('❌ Failed to connect to backend. Please check if the server is running.');
    }
}

// Get a new random object to draw
async function getNewObject() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/random-object`);
        const data = await response.json();
        
        if (data.success) {
            currentObject = data.object;
            const emoji = data.emoji;
            objectPlaceholder.textContent = `${emoji} ${currentObject.charAt(0).toUpperCase() + currentObject.slice(1)}`;
        } else {
            // Fallback to local selection from 21 classes - Updated to match backend
            const objects = [
                'airplane', 'apple', 'banana', 'bicycle', 'bus', 'car', 'cat', 'computer', 'dog', 'elephant', 'fish', 'flower', 'horse', 'house', 'moon', 'rabbit', 'smiley face', 'star', 'sun', 'tree', 'truck'
            ];
            const emojiMap = {
                'airplane': '✈️', 'apple': '🍎', 'banana': '🍌', 'bicycle': '🚲', 'bus': '🚌',
                'car': '�', 'cat': '�', 'computer': '�', 'dog': '🐶', 'elephant': '�',
                'fish': '�', 'flower': '🌸', 'horse': '�', 'house': '�', 'moon': '�',
                'rabbit': '🐰', 'smiley face': '😊', 'star': '⭐', 'sun': '☀️', 'tree': '🌳', 'truck': '🚚'
            };
            currentObject = objects[Math.floor(Math.random() * objects.length)];
            const emoji = emojiMap[currentObject] || '❓';
            objectPlaceholder.textContent = `${emoji} ${currentObject.charAt(0).toUpperCase() + currentObject.slice(1)}`;
        }
    } catch (error) {
        console.error('Error getting random object:', error);
        // Fallback
        currentObject = 'apple';
        objectPlaceholder.textContent = '🍎 Apple';
    }
}

// Event listeners
startButton.addEventListener("click", startGame);
restartButton.addEventListener("click", restartGame);
clearButton.addEventListener("click", clearCanvas);

// Timer function
function startTimer() {
    timer = setInterval(() => {
        timeLeft--;
        timeLeftDisplay.textContent = timeLeft;
        
        // Change color when time is running out
        if (timeLeft <= 10) {
            timeLeftDisplay.style.color = '#ff4444';
        } else if (timeLeft <= 20) {
            timeLeftDisplay.style.color = '#ff8800';
        }
        
        if (timeLeft <= 0) {
            clearInterval(timer);
            endGame();
        }
    }, 1000);
}

// Start the game
function startGame() {
    startScreen.style.display = "none";
    gameScreen.style.display = "block";
    currentObjectDisplay.textContent = objectPlaceholder.textContent;
    timeLeft = 30;
    timeLeftDisplay.textContent = timeLeft;
    timeLeftDisplay.style.color = '#333';
    drawingData = [];
    clearCanvas();
    startTimer();
}

// Drawing event listeners
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseout", stopDrawing);

// Touch events for mobile
canvas.addEventListener("touchstart", handleTouch);
canvas.addEventListener("touchmove", handleTouch);
canvas.addEventListener("touchend", stopDrawing);

function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(e.type.replace('touch', 'mouse'), {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
}

function startDrawing(e) {
    drawing = true;
    [lastX, lastY] = getCoordinates(e);
    
    // Add the starting point to drawing data
    drawingData.push({ x: lastX, y: lastY });
}

function draw(e) {
    if (!drawing) return;
    
    const [x, y] = getCoordinates(e);
    
    // Store the drawing coordinates
    drawingData.push({ x, y });
    
    // Draw on canvas
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    
    [lastX, lastY] = [x, y];
}

function stopDrawing() {
    drawing = false;
}

// Get coordinates for mouse events
function getCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    return [
        e.clientX - rect.left,
        e.clientY - rect.top
    ];
}

// Clear canvas
function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawingData = [];
}

// End game and get prediction
async function endGame() {
    gameScreen.style.display = "none";
    postGameScreen.style.display = "block";
    
    // Show loading message
    modelGuessDisplay.innerHTML = '<div style="color: #666;">🤔 Analyzing your drawing...</div>';
    
    await sendDrawingData();
}

// Send drawing data to backend for recognition
async function sendDrawingData() {
    if (drawingData.length === 0) {
        modelGuessDisplay.innerHTML = `
            <div style="color: #ff4444;">
                <strong>❌ No drawing detected!</strong><br>
                <small>You need to draw something for me to recognize!</small>
            </div>
        `;
        return;
    }

    const requestData = {
        drawing: drawingData,
        object: currentObject
    };

    try {
        const response = await fetch(`${API_BASE_URL}/api/recognize-drawing`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(requestData)
        });

        // Check if the response is ok
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            console.error("Server responded with error:", response.status, errorData);
            
            let errorMessage = `Server error (${response.status})`;
            if (errorData.detail) {
                errorMessage += `: ${JSON.stringify(errorData.detail)}`;
            } else if (errorData.error) {
                errorMessage += `: ${errorData.error}`;
            }
            
            modelGuessDisplay.innerHTML = `
                <div style="color: #ff4444;">
                    <strong>❌ ${errorMessage}</strong><br>
                    <small>Please try drawing again or check the server logs.</small>
                </div>
            `;
            return;
        }

        const data = await response.json();
        console.log("✅ Received response:", data);

        if (data.error) {
            modelGuessDisplay.innerHTML = `
                <div style="color: #ff4444;">
                    <strong>❌ Error:</strong> ${data.error}<br>
                    <small>Please try again or check if the backend is running.</small>
                </div>
            `;
            return;
        }

        // Display comprehensive results
        displayPredictionResults(data);

    } catch (error) {
        console.error("Error sending drawing data:", error);
        modelGuessDisplay.innerHTML = `
            <div style="color: #ff4444;">
                <strong>❌ Network Error</strong><br>
                <small>Could not connect to the AI model: ${error.message}</small>
            </div>
        `;
    }
}

// Display detailed prediction results
function displayPredictionResults(data) {
    console.log("🔍 Raw prediction data received:", data); // Debug log
    
    const prediction = data.prediction;
    const expectedObject = data.expected_object;
    const isCorrect = data.is_correct;
    const confidence = Math.round(data.confidence * 100);
    const topPredictions = data.top_predictions || {};

    console.log("🎯 Processed data:", { prediction, expectedObject, isCorrect, confidence, topPredictions }); // Debug log

    // Get emojis for all 21 classes
    const emojiMap = {
        'airplane': '✈️', 'alarm clock': '⏰', 'apple': '🍎', 'banana': '🍌', 'bicycle': '🚲',
        'bird': '🐦', 'car': '🚗', 'cat': '🐱', 'chair': '🪑', 'clock': '🕐',
        'dog': '🐶', 'elephant': '🐘', 'fish': '🐟', 'flower': '🌸', 'house': '🏠',
        'ice cream': '🍦', 'pencil': '✏️', 'pizza': '🍕', 'spider': '🕷️', 'tree': '🌳', 'umbrella': '☂️'
    };
    
    const predEmoji = emojiMap[prediction] || '❓';
    const expectedEmoji = emojiMap[expectedObject] || '❓';
    const resultEmoji = isCorrect ? '🎉' : '😅';

    console.log("🎨 Emojis:", { predEmoji, expectedEmoji, resultEmoji }); // Debug log

    // Create top predictions HTML with better styling for 21 classes
    let topPredictionsHTML = '';
    const topEntries = Object.entries(topPredictions).slice(0, 3);
    if (topEntries.length > 0) {
        topPredictionsHTML = `
            <div style="background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 15px; border-radius: 12px; margin: 15px 0; border: 1px solid #dee2e6;">
                <div style="font-weight: bold; color: #495057; margin-bottom: 8px;">🏆 AI's Top 3 Guesses:</div>
                ${topEntries.map(([className, conf], index) => {
                    const emoji = emojiMap[className] || '❓';
                    const percentage = Math.round(conf * 100);
                    const medal = index === 0 ? '🥇' : index === 1 ? '🥈' : '🥉';
                    const isWinner = className === prediction;
                    return `
                        <div style="margin: 5px 0; padding: 5px; ${isWinner ? 'background: rgba(76, 175, 80, 0.1); border-radius: 6px; font-weight: bold;' : ''}">
                            ${medal} ${emoji} ${className.charAt(0).toUpperCase() + className.slice(1)}: ${percentage}%
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    } else {
        // Fallback if no top predictions available
        topPredictionsHTML = `
            <div style="background: #fff3cd; padding: 10px; border-radius: 8px; margin: 10px 0; border: 1px solid #ffeaa7;">
                <small>🤖 AI Analysis: Single prediction made</small>
            </div>
        `;
    }

    // Enhanced result HTML - COMPLETELY NEW VERSION
    const resultHTML = `
        <div style="text-align: center; padding: 25px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: linear-gradient(135deg, #f8f9fa, #ffffff); border-radius: 20px;">
            <div style="font-size: 4em; margin-bottom: 15px;">${resultEmoji}</div>
            <h2 style="margin-bottom: 25px; color: ${isCorrect ? '#28a745' : '#6c757d'}; font-size: 2em; font-weight: 700;">
                ${isCorrect ? '🎯 PERFECT MATCH!' : '🎨 NICE TRY!'}
            </h2>
            
            <div style="background: #ffffff; padding: 25px; border-radius: 15px; margin: 20px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border: 2px solid #e9ecef;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                    <div style="text-align: center; padding: 15px; background: #e3f2fd; border-radius: 10px;">
                        <div style="font-weight: bold; color: #1976d2; margin-bottom: 5px;">🎯 TARGET</div>
                        <div style="font-size: 2em;">${expectedEmoji}</div>
                        <div style="font-size: 1.1em; font-weight: 600; color: #1976d2;">${expectedObject.toUpperCase()}</div>
                    </div>
                    <div style="text-align: center; padding: 15px; background: #e8f5e8; border-radius: 10px;">
                        <div style="font-weight: bold; color: #388e3c; margin-bottom: 5px;">🤖 AI GUESS</div>
                        <div style="font-size: 2em;">${predEmoji}</div>
                        <div style="font-size: 1.1em; font-weight: 600; color: #388e3c;">${prediction.toUpperCase()}</div>
                    </div>
                </div>

                <div style="margin: 25px 0;">
                    <div style="font-size: 1.3em; font-weight: bold; margin-bottom: 10px; color: ${confidence > 40 ? '#28a745' : confidence > 20 ? '#ffc107' : '#dc3545'};">
                        🎯 CONFIDENCE: ${confidence}%
                    </div>
                    <div style="background: #e9ecef; height: 12px; border-radius: 6px; overflow: hidden;">
                        <div style="background: ${confidence > 40 ? '#28a745' : confidence > 20 ? '#ffc107' : '#dc3545'}; height: 100%; width: ${confidence}%; transition: width 1s ease; border-radius: 6px;"></div>
                    </div>
                </div>
            </div>

            ${topPredictionsHTML}

            <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; border-radius: 15px; margin: 20px 0;">
                <div style="font-size: 1.1em; line-height: 1.5;">
                    ${isCorrect 
                        ? "🌟 AMAZING! Your drawing was crystal clear!" 
                        : confidence > 30 
                            ? `🎨 Great effort! The AI was ${confidence}% sure. Try emphasizing key features!`
                            : "🤔 Tricky one! Remember to draw the most recognizable parts clearly!"
                    }
                </div>
            </div>

            <div style="background: rgba(102, 126, 234, 0.1); padding: 15px; border-radius: 10px; color: #667eea; font-weight: 600;">
                🎮 QuickDraw Challenge: 21 Object Categories Available!
            </div>
        </div>
    `;

    console.log("✅ Setting result HTML"); // Debug log
    modelGuessDisplay.innerHTML = resultHTML;
    console.log("✅ Result HTML set successfully"); // Debug log
}

// Restart game
async function restartGame() {
    postGameScreen.style.display = "none";
    startScreen.style.display = "block";
    
    // Reset game state
    timeLeft = 30;
    timeLeftDisplay.textContent = timeLeft;
    timeLeftDisplay.style.color = '#333';
    drawingData = [];
    clearCanvas();
    
    // Get a new object to draw
    await getNewObject();
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    if (e.key === 'c' || e.key === 'C') {
        if (gameScreen.style.display === 'block') {
            clearCanvas();
        }
    }
    if (e.key === 'Enter') {
        if (startScreen.style.display !== 'none') {
            startGame();
        } else if (postGameScreen.style.display !== 'none') {
            restartGame();
        }
    }
});

// Prevent scrolling when drawing on mobile
document.body.addEventListener('touchstart', function(e) {
    if (e.target === canvas) {
        e.preventDefault();
    }
}, { passive: false });

document.body.addEventListener('touchend', function(e) {
    if (e.target === canvas) {
        e.preventDefault();
    }
}, { passive: false });

document.body.addEventListener('touchmove', function(e) {
    if (e.target === canvas) {
        e.preventDefault();
    }
}, { passive: false });
