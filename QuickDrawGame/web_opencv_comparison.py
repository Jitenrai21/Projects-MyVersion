#!/usr/bin/env python3
"""
Web vs OpenCV Preprocessing Analysis & Adaptation
===============================================

DETAILED COMPARISON and ADAPTATION STRATEGY

OpenCV Pipeline (WORKING):
  Canvas: (480, 640) → Blur → Threshold → Crop: (109, 109) → Resize: (28, 28) → Raw [0-255]

Current Web Pipeline (BIASED):
  Canvas: (400, 400) → Blur → Resize: (64, 64) → Normalized [0-1]
"""

import sys
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

# Add backend path
sys.path.append('backend/app/models')

def analyze_current_web_preprocessing():
    """
    Analyze current web preprocessing step by step
    """
    print("🌐 CURRENT WEB PREPROCESSING ANALYSIS")
    print("=" * 50)
    
    try:
        from drawing_model import preprocess_drawing_to_image
        
        # Test drawing (square)
        test_drawing = [
            {'x': 150, 'y': 150}, {'x': 250, 'y': 150},
            {'x': 250, 'y': 250}, {'x': 150, 'y': 250}, {'x': 150, 'y': 150}
        ]
        
        print("📊 Current Web Pipeline Steps:")
        print("   1. Canvas: Fixed (400, 400)")
        print("   2. Drawing: Variable stroke width (4-8 pixels)")
        print("   3. Blur: GaussianBlur(radius=0.8)")
        print("   4. Resize: Direct 400x400 → 64x64")
        print("   5. Normalization: Divide by 255 → [0-1] range")
        print("   6. Model: 64x64 model input")
        print("   ❌ Missing: Threshold step")
        print("   ❌ Missing: Cropping step")
        
        # Process the drawing
        result = preprocess_drawing_to_image(test_drawing)
        
        if result is not None:
            print(f"\n📈 Current Web Results:")
            print(f"   Final shape: {result.shape}")
            print(f"   Value range: {result.min():.3f} - {result.max():.3f}")
            print(f"   ⚠️ Values are normalized [0-1], OpenCV uses [0-255]!")
            
            return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return None

def simulate_opencv_preprocessing():
    """
    Simulate OpenCV preprocessing for comparison
    """
    print(f"\n🔍 OPENCV PREPROCESSING SIMULATION")
    print("=" * 50)
    
    # Create OpenCV-style canvas (480, 640)
    opencv_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw same square with 7-pixel lines
    square_points = [(200, 150), (300, 150), (300, 250), (200, 250), (200, 150)]
    for i in range(len(square_points) - 1):
        cv2.line(opencv_canvas, square_points[i], square_points[i+1], (255, 255, 255), 7)
    
    print("📊 OpenCV Pipeline Steps:")
    print("   1. Canvas: Camera (480, 640)")
    print("   2. Convert: BGR → Grayscale")
    print("   3. Blur: MedianBlur(15) + GaussianBlur(5x5)")
    print("   4. Threshold: OTSU binary")
    print("   5. Crop: Find contour → Bounding box")
    print("   6. Resize: Crop → 28x28")
    print("   7. Values: Raw [0-255] (no normalization)")
    
    # Step by step processing
    print(f"\n📈 Step-by-step processing:")
    
    # Convert to grayscale
    gray = cv2.cvtColor(opencv_canvas, cv2.COLOR_BGR2GRAY)
    print(f"   After grayscale: {gray.shape}, range: {gray.min()}-{gray.max()}")
    
    # Blur
    blur1 = cv2.medianBlur(gray, 15)
    blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
    print(f"   After blur: {blur1.shape}, range: {blur1.min()}-{blur1.max()}")
    
    # Threshold
    _, thresh = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"   After threshold: {thresh.shape}, range: {thresh.min()}-{thresh.max()}")
    
    # Find contour and crop
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    if len(contours) >= 1:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = gray[y:y + h, x:x + w]
        print(f"   After crop: {cropped.shape}, range: {cropped.min()}-{cropped.max()}")
        
        # Resize to 28x28
        resized = cv2.resize(cropped, (28, 28))
        print(f"   Final: {resized.shape}, range: {resized.min()}-{resized.max()}")
        
        # Model format
        final = np.array(resized, dtype=np.float32)
        final = final.reshape(-1, 28, 28, 1)
        print(f"   Model input: {final.shape}, range: {final.min():.1f}-{final.max():.1f}")
        
        return final
    
    return None

def compare_preprocessing_methods():
    """
    Direct comparison of both methods
    """
    print(f"\n🔄 PREPROCESSING COMPARISON")
    print("=" * 50)
    
    web_result = analyze_current_web_preprocessing()
    opencv_result = simulate_opencv_preprocessing()
    
    print(f"\n📊 CRITICAL DIFFERENCES:")
    print("=" * 30)
    
    print(f"1. 📐 CANVAS & RESOLUTION:")
    print(f"   OpenCV: 480×640 → Dynamic crop → 28×28")
    print(f"   Web: 400×400 → Direct resize → 64×64")
    print(f"   🚨 DIFFERENT TARGET SIZES!")
    
    print(f"\n2. 🔧 PREPROCESSING STEPS:")
    print(f"   OpenCV: Blur → Threshold → Crop → Resize")
    print(f"   Web: Blur → Resize")
    print(f"   🚨 MISSING: Threshold & Crop in web!")
    
    print(f"\n3. 📊 VALUE RANGES:")
    if web_result is not None and opencv_result is not None:
        print(f"   OpenCV: {opencv_result.min():.1f} - {opencv_result.max():.1f} (raw)")
        print(f"   Web: {web_result.min():.3f} - {web_result.max():.3f} (normalized)")
        print(f"   🚨 DIFFERENT VALUE SCALES!")
    
    print(f"\n4. 🤖 MODELS:")
    print(f"   OpenCV: 28×28 model (models/QuickDraw.h5)")
    print(f"   Web: 64×64 model (QuickDraw_improved_64x64_final.keras)")
    print(f"   🚨 DIFFERENT MODELS = DIFFERENT TRAINING DATA!")

def create_opencv_adapted_web_pipeline():
    """
    Create OpenCV-adapted preprocessing for web system
    """
    print(f"\n🔧 OPENCV-ADAPTED WEB PIPELINE")
    print("=" * 50)
    
    adapted_code = '''
def preprocess_drawing_to_image_opencv_adapted(drawing_data, canvas_size=(400, 400)):
    """
    OpenCV-adapted preprocessing for web system
    Matches QuickDrawApp.py pipeline exactly: Blur → Threshold → Crop → Resize(28x28)
    """
    try:
        if not drawing_data or len(drawing_data) == 0:
            return None
        
        # Step 1: Create canvas and draw (like web system)
        img = Image.new('L', canvas_size, color=0)  # BLACK background
        draw = ImageDraw.Draw(img)
        
        # Step 2: Draw with FIXED 7-pixel strokes (like OpenCV)
        line_width = 7  # FIXED to match OpenCV exactly
        
        # Process strokes (same as current web system)
        strokes = []
        current_stroke = []
        
        for i, point in enumerate(drawing_data):
            if 'strokeEnd' in point:
                if current_stroke:
                    strokes.append(current_stroke)
                    current_stroke = []
                continue
            
            current_stroke.append(point)
            
            if i < len(drawing_data) - 1:
                next_point = drawing_data[i + 1]
                if 'strokeEnd' not in next_point:
                    dist = ((next_point['x'] - point['x'])**2 + (next_point['y'] - point['y'])**2)**0.5
                    if dist > 40:
                        strokes.append(current_stroke)
                        current_stroke = []
        
        if current_stroke:
            strokes.append(current_stroke)
        
        # Draw strokes
        for stroke in strokes:
            if len(stroke) > 1:
                for i in range(len(stroke) - 1):
                    x1, y1 = int(stroke[i]['x']), int(stroke[i]['y'])
                    x2, y2 = int(stroke[i + 1]['x']), int(stroke[i + 1]['y'])
                    draw.line([(x1, y1), (x2, y2)], fill=255, width=line_width)
            elif len(stroke) == 1:
                x, y = int(stroke[0]['x']), int(stroke[0]['y'])
                radius = line_width // 2
                draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], fill=255)
        
        # Step 3: Convert to numpy for OpenCV processing
        img_array = np.array(img, dtype=np.uint8)
        
        # Step 4: Apply OpenCV-style blur (EXACT parameters)
        blur1 = cv2.medianBlur(img_array, 15)
        blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
        
        # Step 5: Apply OpenCV-style threshold (EXACT method)
        _, thresh = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Step 6: Find contour and crop (EXACT OpenCV method)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        if len(contours) >= 1:
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 200:  # Minimum area like OpenCV
                x, y, w, h = cv2.boundingRect(cnt)
                digit = img_array[y:y + h, x:x + w]  # Crop to bounding box
                
                # Step 7: Resize to 28x28 (EXACT OpenCV target)
                resized = cv2.resize(digit, (28, 28))
                
                # Step 8: Format for model (EXACT OpenCV format)
                final = np.array(resized, dtype=np.float32)
                final = np.reshape(final, (-1, 28, 28, 1))
                
                # Step 9: Keep raw values [0-255] - NO NORMALIZATION
                # This matches OpenCV exactly
                
                return final
        
        return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
'''
    
    print("📝 This adapted pipeline:")
    print("✅ Uses same blur parameters as OpenCV")
    print("✅ Includes threshold step (OTSU)")  
    print("✅ Adds contour-based cropping")
    print("✅ Targets 28×28 like OpenCV")
    print("✅ Keeps raw [0-255] values")
    print("✅ Uses 7-pixel strokes")
    
    return adapted_code

def adaptation_strategy():
    """
    Provide adaptation strategy
    """
    print(f"\n💡 ADAPTATION STRATEGY")
    print("=" * 50)
    
    print(f"🎯 PROBLEM: Multiple fundamental differences")
    print(f"   1. Different resolutions (28×28 vs 64×64)")
    print(f"   2. Different models (trained on different data)")
    print(f"   3. Missing preprocessing steps (threshold, crop)")
    print(f"   4. Different value ranges ([0-255] vs [0-1])")
    
    print(f"\n🚀 SOLUTION OPTIONS:")
    print("=" * 20)
    
    print(f"📋 OPTION 1: Full OpenCV Match (RECOMMENDED)")
    print(f"   ✅ Switch to 28×28 model (models/QuickDraw.h5)")
    print(f"   ✅ Add threshold step")
    print(f"   ✅ Add contour cropping")
    print(f"   ✅ Use raw [0-255] values")
    print(f"   ✅ Fix stroke width to 7 pixels")
    print(f"   📈 Expected: Matches OpenCV performance exactly")
    
    print(f"\n📋 OPTION 2: Hybrid Approach")
    print(f"   ✅ Keep 64×64 model but fix preprocessing")
    print(f"   ✅ Add threshold + crop steps")
    print(f"   ✅ Scale crop to 64×64 instead of 28×28")
    print(f"   📈 Expected: Significant improvement")
    
    print(f"\n📋 OPTION 3: Model Retraining")
    print(f"   ✅ Retrain 64×64 model with web-style preprocessing")
    print(f"   ✅ Match training data to web input characteristics")
    print(f"   📈 Expected: Best long-term solution")
    
    print(f"\n🎯 IMMEDIATE ACTION:")
    print(f"   1. Implement Option 1 (use 28×28 model)")
    print(f"   2. Test results")
    print(f"   3. If successful, consider Option 3 for optimization")

def main():
    """
    Main analysis function
    """
    print("🔍 WEB vs OPENCV PREPROCESSING ANALYSIS")
    print("=" * 55)
    
    compare_preprocessing_methods()
    create_opencv_adapted_web_pipeline()
    adaptation_strategy()
    
    print(f"\n📋 SUMMARY:")
    print("=" * 15)
    print("🔍 ROOT CAUSE: Fundamental preprocessing differences")
    print("🎯 KEY INSIGHT: Need threshold + crop + 28×28 target")
    print("🚀 SOLUTION: Implement OpenCV-adapted pipeline")

if __name__ == "__main__":
    main()