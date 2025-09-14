"""
Quick Inference Adaptation - Modify preprocessing to better match training data
Add this to your existing drawing_model.py or use as a patch
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter

def adapt_image_for_training_match(img_array):
    """
    Apply post-processing to match training data characteristics
    This is a quick domain adaptation without changing training data
    """
    
    # Apply slight gaussian blur to simulate training data style
    # Training data has more continuous grayscale values
    img_blurred = gaussian_filter(img_array, sigma=0.7)
    
    # Increase contrast slightly to make strokes more prominent
    img_contrasted = np.clip(img_blurred * 1.3, 0, 1)
    
    # Apply slight noise to simulate training data artifacts
    noise = np.random.normal(0, 0.02, img_contrasted.shape)
    img_noisy = np.clip(img_contrasted + noise, 0, 1)
    
    return img_noisy

def preprocess_drawing_to_image_adapted(drawing_data, target_size=(28, 28)):
    """
    Enhanced preprocessing that adapts the inference pipeline to better match training data
    This is a drop-in replacement for your existing preprocess_drawing_to_image function
    """
    
    if not drawing_data:
        return None
    
    try:
        # Extract coordinates
        coordinates = [(point["x"], point["y"]) for point in drawing_data]
        
        if len(coordinates) < 2:
            return None
        
        # Create image on 400x400 canvas (square, as implemented)
        canvas_size = 400
        img = Image.new('L', (canvas_size, canvas_size), 0)
        draw = ImageDraw.Draw(img)
        
        # Separate strokes (detect pen lifts)
        strokes = []
        current_stroke = [coordinates[0]]
        
        for i in range(1, len(coordinates)):
            prev_point = coordinates[i-1]
            curr_point = coordinates[i]
            
            # Calculate distance between consecutive points
            distance = ((curr_point[0] - prev_point[0])**2 + (curr_point[1] - prev_point[1])**2)**0.5
            
            # If distance is too large, it's likely a new stroke
            if distance > 50:  # Threshold for stroke separation
                strokes.append(current_stroke)
                current_stroke = [curr_point]
            else:
                current_stroke.append(curr_point)
        
        if current_stroke:
            strokes.append(current_stroke)
        
        # Draw each stroke
        stroke_width = 3  # Slightly thicker to match training data
        for stroke in strokes:
            if len(stroke) > 1:
                # Draw lines between consecutive points in stroke
                for j in range(len(stroke) - 1):
                    draw.line([stroke[j], stroke[j + 1]], fill=255, width=stroke_width)
            elif len(stroke) == 1:
                # Draw a single point
                x, y = stroke[0]
                draw.ellipse([x-1, y-1, x+1, y+1], fill=255)
        
        # Apply slight blur to anti-alias (similar to training data)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Resize to target size with proper aspect ratio preservation
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        
        # Apply domain adaptation post-processing
        img_array = adapt_image_for_training_match(img_array)
        
        # Reshape for model input
        img_array = img_array.reshape(1, target_size[0], target_size[1], 1)
        
        return img_array
        
    except Exception as e:
        print(f"❌ Error in adapted preprocessing: {e}")
        return None

# Patch instructions for your existing drawing_model.py:
"""
To apply this adaptation:

1. Add the adapt_image_for_training_match function to your drawing_model.py

2. In your existing preprocess_drawing_to_image function, add this line after converting to numpy:
   
   # Apply domain adaptation (ADD THIS LINE)
   img_array = adapt_image_for_training_match(img_array)

3. Or replace your entire preprocess_drawing_to_image function with preprocess_drawing_to_image_adapted

This should improve domain matching by:
- Adding slight blur to simulate training data characteristics
- Increasing contrast for better stroke definition  
- Adding minimal noise to match training data artifacts
- Using slightly thicker strokes
- Better anti-aliasing
"""