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
const predictionTextDisplay = document.getElementById("prediction-text");
const canvas = document.getElementById("drawing-canvas");
const ctx = canvas.getContext("2d");

// Game state variables
let drawing = false;
let lastX = 0;
let lastY = 0;
let timeLeft = 30;
let timer;
let currentObject = "";
let gameActive = false;
let gameWon = false;

// Real-time evaluation variables
let evaluationTimeout;
let lastEvaluationTime = 0;
const EVALUATION_DELAY = 1000; // 1 second delay between evaluations
let isEvaluating = false;

// Array to hold the sequence of drawing coordinates with stroke information
// Updated for square canvas (400x400)
let drawingData = [];
let currentStroke = []; // Track current stroke
let strokeStartTime = 0;
const CANVAS_SIZE = { width: 400, height: 400 }; // Square canvas constants

// Set canvas size - Square canvas for better aspect ratio
canvas.width = 400;
canvas.height = 400;

// Configure canvas for better drawing
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = '#000';
ctx.lineWidth = 5;

// API base URL - adjust if your backend runs on different port
const API_BASE_URL = window.location.origin.includes('localhost') ? 
    'http://localhost:8000' : window.location.origin;

// Global emoji mapping for all objects
const emojiMap = {
    'apple': '🍎', 'bowtie': '🎀', 'candle': '🕯️', 'door': '🚪', 'envelope': '✉️',
    'fish': '🐟', 'guitar': '🎸', 'ice cream': '🍦', 'lightning': '⚡', 'moon': '🌙',
    'mountain': '⛰️', 'star': '⭐', 'tent': '⛺', 'toothbrush': '🪥', 'wristwatch': '⌚'
};

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
            // Fallback to local selection from 15 classes - Updated to match backend
            const objects = [
                'apple', 'bowtie', 'candle', 'door', 'envelope', 'fish', 'guitar', 'ice cream', 'lightning', 'moon',
                'mountain', 'star', 'tent', 'toothbrush', 'wristwatch'
            ];
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
        // Don't continue timer if game was won early
        if (gameWon || !gameActive) {
            clearInterval(timer);
            return;
        }
        
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
            gameActive = false;
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
    gameActive = true;
    gameWon = false;
    isEvaluating = false;
    
    // Reset prediction display
    if (predictionTextDisplay) {
        predictionTextDisplay.textContent = "Start drawing...";
    }
    
    clearCanvas();
    startTimer();
}

// Real-time drawing evaluation with debouncing
async function evaluateDrawingRealTime() {
    // Don't evaluate if game is not active or already won
    if (!gameActive || gameWon || isEvaluating) return;
    
    // Don't evaluate if there's not enough drawing data
    if (drawingData.length < 10) return;
    
    // Debouncing: don't evaluate too frequently
    const now = Date.now();
    if (now - lastEvaluationTime < EVALUATION_DELAY) return;
    
    lastEvaluationTime = now;
    isEvaluating = true;
    
    try {
        const requestData = {
            drawing: drawingData,
            object: currentObject
        };

        const response = await fetch(`${API_BASE_URL}/api/recognize-drawing`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            console.warn("Real-time evaluation failed:", response.status);
            return;
        }

        const data = await response.json();
        
        if (data.error) {
            console.warn("Real-time evaluation error:", data.error);
            return;
        }

        // NEW SUCCESS LOGIC: Check if highest confidence prediction matches current object
        const highestPrediction = data.prediction.toLowerCase();
        const targetObject = currentObject.toLowerCase();
        const isCorrectPrediction = highestPrediction === targetObject;
        
        // Update AI prediction display (object name only, no confidence)
        if (predictionTextDisplay && gameActive) {
            const emoji = emojiMap[highestPrediction] || '🤔';
            const capitalizedPrediction = highestPrediction.charAt(0).toUpperCase() + highestPrediction.slice(1);
            const displayText = `${emoji} ${capitalizedPrediction}`;
            predictionTextDisplay.textContent = displayText;
            console.log('AI prediction updated:', displayText); // Debug log
        }
        
        if (isCorrectPrediction) {
            // SUCCESS! Highest confidence prediction matches target
            gameWon = true;
            gameActive = false;
            const actualTime = 30 - timeLeft;
            clearInterval(timer);
            showImmediateSuccess(data, actualTime);
        }
        
    } catch (error) {
        console.warn("Real-time evaluation network error:", error);
    } finally {
        isEvaluating = false;
    }
}

// Show immediate success screen
function showImmediateSuccess(data, actualTime) {
    gameScreen.style.display = "none";
    postGameScreen.style.display = "block";
    
    // Get emoji from global mapping
    const emoji = emojiMap[currentObject] || '❓';
    
    const successHTML = `
        <div style="text-align: center; padding: 40px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: linear-gradient(135deg, #28a745, #20c997); border-radius: 20px; color: white;">
            <div style="font-size: 6em; margin-bottom: 30px;">🎉</div>
            <h1 style="margin-bottom: 20px; font-size: 3em; font-weight: 700;">
                AMAZING!
            </h1>
            <h2 style="margin-bottom: 30px; font-size: 2em; opacity: 0.9;">
                AI Recognized Your ${emoji} ${currentObject.toUpperCase()}!
            </h2>
            
            <div style="background: rgba(255,255,255,0.15); padding: 20px; border-radius: 15px; margin: 20px 0;">
                <div style="font-size: 1.3em; line-height: 1.6;">
                    🚀 Perfect! The AI recognized your drawing!<br>
                    Great job on your artistic skills! 🎨
                </div>
            </div>
        </div>
    `;
    
    modelGuessDisplay.innerHTML = successHTML;
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
    strokeStartTime = Date.now();
    
    // Start a new stroke
    currentStroke = [{ x: lastX, y: lastY, timestamp: strokeStartTime }];
}

function draw(e) {
    if (!drawing) return;
    
    const [x, y] = getCoordinates(e);
    const currentTime = Date.now();
    
    // Add point to current stroke
    currentStroke.push({ x, y, timestamp: currentTime });
    
    // Draw on canvas
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    
    [lastX, lastY] = [x, y];
}

function stopDrawing() {
    if (drawing && currentStroke.length > 0) {
        // Add the completed stroke to drawing data
        drawingData = drawingData.concat(currentStroke);
        
        // Add a small gap indicator for stroke separation
        if (currentStroke.length > 1) {
            const lastPoint = currentStroke[currentStroke.length - 1];
            // Add a point far away to indicate stroke end
            drawingData.push({ 
                x: lastPoint.x + 100, 
                y: lastPoint.y + 100, 
                timestamp: Date.now(),
                strokeEnd: true 
            });
        }
        
        currentStroke = [];
        
        // Trigger real-time evaluation after each stroke completion
        if (gameActive && !gameWon) {
            // Clear any pending evaluation
            if (evaluationTimeout) {
                clearTimeout(evaluationTimeout);
            }
            
            // Schedule evaluation with slight delay to allow for multi-stroke drawings
            evaluationTimeout = setTimeout(() => {
                evaluateDrawingRealTime();
            }, 500); // 500ms delay after stroke completion
        }
    }
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
    currentStroke = [];
    
    // Reset prediction display when clearing
    if (predictionTextDisplay && gameActive) {
        predictionTextDisplay.textContent = "Start drawing...";
    }
    
    // Clear any pending evaluations
    if (evaluationTimeout) {
        clearTimeout(evaluationTimeout);
        evaluationTimeout = null;
    }
}

// End game and get prediction
async function endGame() {
    // If game was already won through real-time recognition, don't process again
    if (gameWon) {
        return;
    }
    
    gameActive = false;
    gameScreen.style.display = "none";
    postGameScreen.style.display = "block";
    
    // Show loading message
    modelGuessDisplay.innerHTML = '<div style="color: #666;">🤔 Analyzing your final drawing...</div>';
    
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

// Display simplified prediction results
function displayPredictionResults(data) {
    const prediction = data.prediction;
    const expectedObject = data.expected_object;
    const isCorrect = data.is_correct;

    // Get emojis from global mapping
    const predEmoji = emojiMap[prediction] || '❓';
    const expectedEmoji = emojiMap[expectedObject] || '❓';
    const resultEmoji = isCorrect ? '🎉' : '😅';

    // Simple and clean result HTML
    const resultHTML = `
        <div style="text-align: center; padding: 40px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: linear-gradient(135deg, ${isCorrect ? '#28a745, #20c997' : '#6c757d, #495057'}); border-radius: 20px; color: white;">
            <div style="font-size: 6em; margin-bottom: 30px;">${resultEmoji}</div>
            <h2 style="margin-bottom: 30px; font-size: 3em; font-weight: 700;">
                ${isCorrect ? 'PERFECT!' : 'NICE TRY!'}
            </h2>
            
            <div style="background: rgba(255,255,255,0.15); padding: 25px; border-radius: 15px; margin: 25px 0;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px;">
                    <div style="text-align: center;">
                        <div style="font-weight: bold; margin-bottom: 10px; font-size: 1.1em;">🎯 YOU DREW</div>
                        <div style="font-size: 3em; margin-bottom: 10px;">${expectedEmoji}</div>
                        <div style="font-size: 1.3em; font-weight: 600;">${expectedObject.toUpperCase()}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-weight: bold; margin-bottom: 10px; font-size: 1.1em;">🤖 AI SAW</div>
                        <div style="font-size: 3em; margin-bottom: 10px;">${predEmoji}</div>
                        <div style="font-size: 1.3em; font-weight: 600;">${prediction.toUpperCase()}</div>
                    </div>
                </div>
            </div>

            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; margin: 20px 0;">
                <div style="font-size: 1.3em; line-height: 1.6;">
                    ${isCorrect 
                        ? "🌟 Excellent drawing! The AI recognized it perfectly!" 
                        : "🎨 Keep practicing! Try to emphasize the key features next time!"
                    }
                </div>
            </div>
        </div>
    `;

    modelGuessDisplay.innerHTML = resultHTML;
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
    gameActive = false;
    gameWon = false;
    isEvaluating = false;
    
    // Reset prediction display
    if (predictionTextDisplay) {
        predictionTextDisplay.textContent = "Start drawing...";
    }
    
    // Clear any pending evaluations
    if (evaluationTimeout) {
        clearTimeout(evaluationTimeout);
        evaluationTimeout = null;
    }
    
    // Hide confidence display
    const confidenceDisplay = document.getElementById('confidence-display');
    if (confidenceDisplay) {
        confidenceDisplay.style.opacity = '0';
    }
    
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
