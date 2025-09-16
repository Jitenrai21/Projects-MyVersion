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
let gameActive = false;
let gameWon = false;

// Real-time evaluation variables
let evaluationTimeout;
let lastEvaluationTime = 0;
const EVALUATION_DELAY = 1000; // 1 second delay between evaluations
const SUCCESS_THRESHOLD = 0.7; // 70% confidence threshold for immediate success
let currentConfidence = 0;
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
            const emojiMap = {
                'apple': '🍎', 'bowtie': '🎀', 'candle': '🕯️', 'door': '🚪', 'envelope': '✉️',
                'fish': '🐟', 'guitar': '🎸', 'ice cream': '🍦', 'lightning': '⚡', 'moon': '🌙',
                'mountain': '⛰️', 'star': '⭐', 'tent': '⛺', 'toothbrush': '🪥', 'wristwatch': '⌚'
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
    currentConfidence = 0;
    isEvaluating = false;
    clearCanvas();
    startTimer();
    
    // Add real-time confidence display if it doesn't exist
    if (!document.getElementById('confidence-display')) {
        addConfidenceDisplay();
    }
}

// Add real-time confidence display to the UI
function addConfidenceDisplay() {
    const gameScreen = document.getElementById('game-screen');
    const confidenceDiv = document.createElement('div');
    confidenceDiv.id = 'confidence-display';
    confidenceDiv.style.cssText = `
        position: absolute;
        top: 60px;
        right: 20px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 10px 15px;
        border-radius: 10px;
        font-weight: bold;
        min-width: 150px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        opacity: 0;
        transition: opacity 0.3s ease;
    `;
    confidenceDiv.innerHTML = `
        <div style="font-size: 0.8em; margin-bottom: 2px;">🤖 AI Confidence</div>
        <div id="confidence-value" style="font-size: 1.2em;">0%</div>
        <div id="confidence-status" style="font-size: 0.7em; margin-top: 2px;">Keep drawing...</div>
    `;
    gameScreen.appendChild(confidenceDiv);
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

        // Update confidence display
        currentConfidence = data.confidence;
        updateConfidenceDisplay(data);
        
        // Check for success condition
        const isCorrect = data.prediction.toLowerCase() === currentObject.toLowerCase();
        const highConfidence = data.confidence >= SUCCESS_THRESHOLD;
        
        if (isCorrect && highConfidence) {
            // IMMEDIATE SUCCESS!
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

// Update the confidence display with real-time feedback
function updateConfidenceDisplay(data) {
    const confidenceDisplay = document.getElementById('confidence-display');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceStatus = document.getElementById('confidence-status');
    
    if (!confidenceDisplay || !confidenceValue || !confidenceStatus) return;
    
    const confidence = Math.round(data.confidence * 100);
    const isCorrect = data.prediction.toLowerCase() === currentObject.toLowerCase();
    
    // Show the display
    confidenceDisplay.style.opacity = '1';
    
    // Update confidence value
    confidenceValue.textContent = `${confidence}%`;
    
    // Update status and styling based on correctness and confidence
    if (isCorrect) {
        if (confidence >= SUCCESS_THRESHOLD * 100) {
            confidenceStatus.textContent = "🎉 RECOGNIZED!";
            confidenceDisplay.style.background = "linear-gradient(135deg, #28a745, #20c997)";
        } else {
            confidenceStatus.textContent = `✅ ${data.prediction} (${SUCCESS_THRESHOLD * 100}% needed)`;
            confidenceDisplay.style.background = "linear-gradient(135deg, #ffc107, #fd7e14)";
        }
    } else {
        confidenceStatus.textContent = `🤔 Sees: ${data.prediction}`;
        confidenceDisplay.style.background = "linear-gradient(135deg, #6c757d, #495057)";
    }
}

// Show immediate success screen
function showImmediateSuccess(data, actualTime) {
    gameScreen.style.display = "none";
    postGameScreen.style.display = "block";
    
    const confidence = Math.round(data.confidence * 100);
    
    // Get emojis
    const emojiMap = {
        'apple': '🍎', 'bowtie': '🎀', 'candle': '🕯️', 'door': '🚪', 'envelope': '✉️',
        'fish': '🐟', 'guitar': '🎸', 'ice cream': '🍦', 'lightning': '⚡', 'moon': '🌙',
        'mountain': '⛰️', 'star': '⭐', 'tent': '⛺', 'toothbrush': '🪥', 'wristwatch': '⌚'
    };
    
    const emoji = emojiMap[currentObject] || '❓';
    
    const successHTML = `
        <div style="text-align: center; padding: 30px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: linear-gradient(135deg, #28a745, #20c997); border-radius: 20px; color: white;">
            <div style="font-size: 5em; margin-bottom: 20px;">🎉</div>
            <h1 style="margin-bottom: 15px; font-size: 2.5em; font-weight: 700;">
                AMAZING!
            </h1>
            <h2 style="margin-bottom: 25px; font-size: 1.8em; opacity: 0.9;">
                AI Recognized Your ${emoji} ${currentObject.toUpperCase()}!
            </h2>
            
            <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 15px; margin: 20px 0;">
                <div style="font-size: 1.5em; font-weight: bold; margin-bottom: 10px;">
                    ⚡ INSTANT RECOGNITION!
                </div>
                <div style="font-size: 1.2em; margin-bottom: 10px;">
                    🎯 Confidence: ${confidence}% (needed ${SUCCESS_THRESHOLD * 100}%)
                </div>
                <div style="font-size: 1.2em;">
                    ⏱️ Time: ${actualTime.toFixed(1)} seconds
                </div>
            </div>

            <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 10px; margin: 20px 0;">
                <div style="font-size: 1.1em; line-height: 1.5;">
                    🚀 Perfect! The AI recognized your drawing immediately!<br>
                    This is exactly how the real QuickDraw game works!
                </div>
            </div>

            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; color: rgba(255,255,255,0.9); font-weight: 600;">
                🎮 HYBRID AI: Real-time Recognition Powered by OpenCV + 64x64 Neural Network
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
    
    // Reset confidence display when clearing
    const confidenceDisplay = document.getElementById('confidence-display');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceStatus = document.getElementById('confidence-status');
    
    if (confidenceDisplay && gameActive) {
        confidenceDisplay.style.opacity = '0';
    }
    if (confidenceValue) {
        confidenceValue.textContent = '0%';
    }
    if (confidenceStatus) {
        confidenceStatus.textContent = 'Keep drawing...';
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

// Display detailed prediction results
function displayPredictionResults(data) {
    console.log("🔍 Raw prediction data received:", data); // Debug log
    
    const prediction = data.prediction;
    const expectedObject = data.expected_object;
    const isCorrect = data.is_correct;
    const confidence = Math.round(data.confidence * 100);
    const topPredictions = data.top_predictions || {};

    console.log("🎯 Processed data:", { prediction, expectedObject, isCorrect, confidence, topPredictions }); // Debug log

    // Get emojis for all 15 classes
    const emojiMap = {
        'apple': '🍎', 'bowtie': '🎀', 'candle': '🕯️', 'door': '🚪', 'envelope': '✉️',
        'fish': '🐟', 'guitar': '🎸', 'ice cream': '🍦', 'lightning': '⚡', 'moon': '🌙',
        'mountain': '⛰️', 'star': '⭐', 'tent': '⛺', 'toothbrush': '🪥', 'wristwatch': '⌚'
    };
    
    const predEmoji = emojiMap[prediction] || '❓';
    const expectedEmoji = emojiMap[expectedObject] || '❓';
    const resultEmoji = isCorrect ? '🎉' : '😅';

    console.log("🎨 Emojis:", { predEmoji, expectedEmoji, resultEmoji }); // Debug log

    // Create top predictions HTML with better styling for 15 classes
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

    // Enhanced result HTML - HYBRID APPROACH VERSION
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

            <div style="background: rgba(76, 175, 80, 0.1); padding: 15px; border-radius: 10px; color: #388e3c; font-weight: 600; margin: 15px 0;">
                🚀 HYBRID AI: OpenCV + 64x64 Neural Network
            </div>
            
            <div style="background: rgba(102, 126, 234, 0.1); padding: 15px; border-radius: 10px; color: #667eea; font-weight: 600;">
                🎮 QuickDraw Challenge: 15 Object Categories Available!
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
    gameActive = false;
    gameWon = false;
    currentConfidence = 0;
    isEvaluating = false;
    
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
