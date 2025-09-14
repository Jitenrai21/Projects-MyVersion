#!/usr/bin/env python3
"""
Aspect Ratio and Coordinate Analysis Script
This script will identify issues with aspect ratio distortion and coordinate handling
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Add the backend directory to the Python path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend', 'app')
sys.path.insert(0, backend_dir)

try:
    from models.drawing_model import preprocess_drawing_to_image, CLASS_LABELS, predict_drawing
    print("✅ Successfully imported drawing model")
except ImportError as e:
    print(f"❌ Failed to import drawing model: {e}")
    sys.exit(1)

def analyze_aspect_ratio_distortion():
    """Analyze how aspect ratio conversion distorts drawings"""
    
    print("\n🔍 ASPECT RATIO DISTORTION ANALYSIS")
    print("=" * 60)
    
    # Test cases showing the problem
    test_cases = {
        "Perfect Circle": create_perfect_circle(),
        "Horizontal Rectangle": create_horizontal_rectangle(), 
        "Vertical Rectangle": create_vertical_rectangle(),
        "Star Shape": create_star_shape(),
        "Moon Crescent": create_moon_crescent()
    }
    
    fig, axes = plt.subplots(len(test_cases), 4, figsize=(16, 4*len(test_cases)))
    fig.suptitle("Aspect Ratio Distortion Analysis", fontsize=16)
    
    for i, (name, coords) in enumerate(test_cases.items()):
        print(f"\n📐 Testing: {name}")
        
        # Get coordinate ranges
        x_coords = [p['x'] for p in coords]
        y_coords = [p['y'] for p in coords]
        
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        aspect_ratio = x_range / y_range if y_range > 0 else 1.0
        
        print(f"   Original aspect ratio: {aspect_ratio:.2f} (width/height)")
        print(f"   X range: {x_range:.1f}px, Y range: {y_range:.1f}px")
        
        # Plot 1: Original coordinates
        axes[i, 0].set_title(f"{name}\n(Original 600x400)")
        axes[i, 0].set_xlim(0, 600)
        axes[i, 0].set_ylim(400, 0)
        axes[i, 0].plot(x_coords, y_coords, 'b-', linewidth=2, marker='o', markersize=2)
        axes[i, 0].set_aspect('equal')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot 2: Current method (distorted)
        processed_distorted = preprocess_drawing_to_image(coords, (600, 400), (28, 28))
        axes[i, 1].set_title("Current Method\n(Distorted to 28x28)")
        if processed_distorted is not None:
            axes[i, 1].imshow(processed_distorted[0, :, :, 0], cmap='gray')
        axes[i, 1].axis('off')
        
        # Plot 3: Aspect-ratio preserved method
        processed_preserved = preprocess_with_aspect_ratio(coords, (600, 400), (28, 28))
        axes[i, 2].set_title("Aspect-Ratio Preserved\n(Padded to 28x28)")
        if processed_preserved is not None:
            axes[i, 2].imshow(processed_preserved[0, :, :, 0], cmap='gray')
        axes[i, 2].axis('off')
        
        # Plot 4: Square canvas method
        processed_square = preprocess_drawing_to_image(
            scale_to_square_canvas(coords, (600, 400)), (400, 400), (28, 28)
        )
        axes[i, 3].set_title("Square Canvas\n(Rescaled coordinates)")
        if processed_square is not None:
            axes[i, 3].imshow(processed_square[0, :, :, 0], cmap='gray')
        axes[i, 3].axis('off')
        
        # Test predictions
        print(f"   Testing predictions...")
        
        if processed_distorted is not None:
            pred_distorted = predict_drawing(coords)
            print(f"   🔴 Current (distorted): {pred_distorted.get('prediction', 'error')} "
                  f"({pred_distorted.get('confidence', 0)*100:.1f}%)")
        
        if processed_preserved is not None:
            pred_preserved = predict_drawing_with_preserved_aspect(coords)
            print(f"   🟢 Aspect preserved: {pred_preserved.get('prediction', 'error')} "
                  f"({pred_preserved.get('confidence', 0)*100:.1f}%)")
    
    plt.tight_layout()
    plt.savefig('aspect_ratio_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_perfect_circle():
    """Create a perfect circle that should remain circular"""
    coords = []
    center_x, center_y = 300, 200
    radius = 80
    for i in range(50):
        angle = (i / 50) * 2 * np.pi
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        coords.append({"x": x, "y": y})
    return coords

def create_horizontal_rectangle():
    """Create a horizontal rectangle that will be distorted"""
    coords = []
    points = [(150, 180), (450, 180), (450, 220), (150, 220), (150, 180)]
    for x, y in points:
        coords.append({"x": x, "y": y})
    return coords

def create_vertical_rectangle():
    """Create a vertical rectangle that will be compressed"""
    coords = []
    points = [(280, 100), (320, 100), (320, 300), (280, 300), (280, 100)]
    for x, y in points:
        coords.append({"x": x, "y": y})
    return coords

def create_star_shape():
    """Create a star that should maintain its proportions"""
    coords = []
    center_x, center_y = 300, 200
    outer_radius = 60
    inner_radius = 30
    for i in range(11):
        angle = (i / 10) * 2 * np.pi
        if i % 2 == 0:
            x = center_x + outer_radius * np.cos(angle - np.pi/2)
            y = center_y + outer_radius * np.sin(angle - np.pi/2)
        else:
            x = center_x + inner_radius * np.cos(angle - np.pi/2)
            y = center_y + inner_radius * np.sin(angle - np.pi/2)
        coords.append({"x": x, "y": y})
    return coords

def create_moon_crescent():
    """Create a moon crescent shape"""
    coords = []
    # Outer arc (left side of moon)
    center_x, center_y = 300, 200
    radius = 60
    for i in range(25):
        angle = np.pi * (0.3 + 1.4 * i / 24)  # From 0.3π to 1.7π
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        coords.append({"x": x, "y": y})
    
    # Inner arc (right side cutout)
    center_x2 = center_x + 20
    radius2 = 45
    for i in range(25):
        angle = np.pi * (1.7 - 1.4 * i / 24)  # From 1.7π to 0.3π (reverse)
        x = center_x2 + radius2 * np.cos(angle)
        y = center_y + radius2 * np.sin(angle)
        coords.append({"x": x, "y": y})
    
    return coords

def preprocess_with_aspect_ratio(drawing_data, canvas_size=(600, 400), target_size=(28, 28)):
    """
    Improved preprocessing that preserves aspect ratio
    """
    try:
        if not drawing_data or len(drawing_data) == 0:
            return None
        
        # Create image with original aspect ratio
        img = Image.new('L', canvas_size, color=255)
        draw = ImageDraw.Draw(img)
        
        # Draw the shapes (using improved stroke separation)
        strokes = separate_strokes(drawing_data)
        line_width = max(2, min(4, int(canvas_size[0] / 150)))
        
        for stroke in strokes:
            if len(stroke) > 1:
                for i in range(len(stroke) - 1):
                    x1, y1 = int(stroke[i]['x']), int(stroke[i]['y'])
                    x2, y2 = int(stroke[i + 1]['x']), int(stroke[i + 1]['y'])
                    draw.line([(x1, y1), (x2, y2)], fill=0, width=line_width)
            elif len(stroke) == 1:
                x, y = int(stroke[0]['x']), int(stroke[0]['y'])
                radius = line_width // 2
                draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], fill=0)
        
        # Calculate aspect ratio
        canvas_ratio = canvas_size[0] / canvas_size[1]  # 600/400 = 1.5
        target_ratio = target_size[0] / target_size[1]  # 28/28 = 1.0
        
        if canvas_ratio != target_ratio:
            # Need to pad to preserve aspect ratio
            if canvas_ratio > target_ratio:
                # Canvas is wider, add padding to height
                new_height = int(canvas_size[0] / target_ratio)
                padding = (new_height - canvas_size[1]) // 2
                new_img = Image.new('L', (canvas_size[0], new_height), color=255)
                new_img.paste(img, (0, padding))
                img = new_img
            else:
                # Canvas is taller, add padding to width
                new_width = int(canvas_size[1] * target_ratio)
                padding = (new_width - canvas_size[0]) // 2
                new_img = Image.new('L', (new_width, canvas_size[1]), color=255)
                new_img.paste(img, (padding, 0))
                img = new_img
        
        # Now resize to target size
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, target_size[0], target_size[1], 1)
        
        return img_array
        
    except Exception as e:
        print(f"❌ Error in aspect-ratio preprocessing: {e}")
        return None

def scale_to_square_canvas(drawing_data, original_canvas_size):
    """
    Scale coordinates to a square canvas to avoid aspect ratio issues
    """
    if not drawing_data:
        return drawing_data
    
    # Use the larger dimension as the square size
    square_size = max(original_canvas_size)
    
    # Calculate scaling and offset
    x_scale = square_size / original_canvas_size[0]
    y_scale = square_size / original_canvas_size[1]
    
    # Calculate centering offsets
    x_offset = (square_size - original_canvas_size[0] * x_scale) / 2
    y_offset = (square_size - original_canvas_size[1] * y_scale) / 2
    
    scaled_coords = []
    for point in drawing_data:
        if 'strokeEnd' in point:
            continue  # Skip stroke end markers
        
        new_x = point['x'] * x_scale + x_offset
        new_y = point['y'] * y_scale + y_offset
        scaled_coords.append({"x": new_x, "y": new_y})
    
    return scaled_coords

def separate_strokes(drawing_data, gap_threshold=50):
    """
    Separate drawing data into individual strokes
    """
    if not drawing_data:
        return []
    
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
                if dist > gap_threshold:
                    strokes.append(current_stroke)
                    current_stroke = []
    
    if current_stroke:
        strokes.append(current_stroke)
    
    return strokes

def predict_drawing_with_preserved_aspect(drawing_data):
    """
    Predict using aspect-ratio preserved preprocessing
    """
    processed_image = preprocess_with_aspect_ratio(drawing_data)
    if processed_image is None:
        return {"error": "Preprocessing failed", "prediction": "unknown", "confidence": 0.0}
    
    # Import the model for prediction
    try:
        from models.drawing_model import model, CLASS_LABELS
        if model is None:
            return {"error": "Model not loaded", "prediction": "unknown", "confidence": 0.0}
        
        prediction_probs = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(prediction_probs[0])
        confidence = float(prediction_probs[0][predicted_class_idx])
        predicted_label = CLASS_LABELS[predicted_class_idx]
        
        # Get top 3 predictions
        top_indices = np.argsort(prediction_probs[0])[-3:][::-1]
        top_predictions = {}
        for idx in top_indices:
            top_predictions[CLASS_LABELS[idx]] = float(prediction_probs[0][idx])
        
        return {
            "prediction": predicted_label,
            "confidence": confidence,
            "top_predictions": top_predictions
        }
        
    except Exception as e:
        return {"error": str(e), "prediction": "unknown", "confidence": 0.0}

def test_moon_bias():
    """
    Specifically test why moon predictions are biased
    """
    print("\n🌙 MOON BIAS ANALYSIS")
    print("=" * 40)
    
    # Create different moon-like shapes
    moon_variants = {
        "Thick Crescent": create_thick_moon_crescent(),
        "Thin Crescent": create_thin_moon_crescent(),
        "Full Circle": create_perfect_circle(),
        "Half Circle": create_half_circle()
    }
    
    for name, coords in moon_variants.items():
        print(f"\n🔍 Testing: {name}")
        
        # Test with current method
        pred_current = predict_drawing(coords)
        print(f"   Current method: {pred_current.get('prediction', 'error')} "
              f"({pred_current.get('confidence', 0)*100:.1f}%)")
        
        # Test with improved method
        pred_improved = predict_drawing_with_preserved_aspect(coords)
        print(f"   Improved method: {pred_improved.get('prediction', 'error')} "
              f"({pred_improved.get('confidence', 0)*100:.1f}%)")
        
        # Show top 3 predictions
        if 'top_predictions' in pred_improved:
            print("   Top 3 predictions (improved):")
            for i, (class_name, conf) in enumerate(list(pred_improved['top_predictions'].items())[:3]):
                print(f"      {i+1}. {class_name}: {conf*100:.1f}%")

def create_thick_moon_crescent():
    """Create a thick moon crescent"""
    coords = []
    center_x, center_y = 300, 200
    
    # Outer arc
    radius1 = 70
    for i in range(30):
        angle = np.pi * (0.2 + 1.6 * i / 29)
        x = center_x + radius1 * np.cos(angle)
        y = center_y + radius1 * np.sin(angle)
        coords.append({"x": x, "y": y})
    
    # Inner arc
    center_x2 = center_x + 25
    radius2 = 50
    for i in range(30):
        angle = np.pi * (1.8 - 1.6 * i / 29)
        x = center_x2 + radius2 * np.cos(angle)
        y = center_y + radius2 * np.sin(angle)
        coords.append({"x": x, "y": y})
    
    return coords

def create_thin_moon_crescent():
    """Create a thin moon crescent"""
    coords = []
    center_x, center_y = 300, 200
    
    # Outer arc
    radius1 = 60
    for i in range(25):
        angle = np.pi * (0.4 + 1.2 * i / 24)
        x = center_x + radius1 * np.cos(angle)
        y = center_y + radius1 * np.sin(angle)
        coords.append({"x": x, "y": y})
    
    # Inner arc (closer to outer)
    center_x2 = center_x + 15
    radius2 = 50
    for i in range(25):
        angle = np.pi * (1.6 - 1.2 * i / 24)
        x = center_x2 + radius2 * np.cos(angle)
        y = center_y + radius2 * np.sin(angle)
        coords.append({"x": x, "y": y})
    
    return coords

def create_half_circle():
    """Create a half circle (D shape)"""
    coords = []
    center_x, center_y = 300, 200
    radius = 60
    
    # Half circle arc
    for i in range(25):
        angle = np.pi * i / 24  # 0 to π
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        coords.append({"x": x, "y": y})
    
    # Straight line to close
    coords.append({"x": center_x - radius, "y": center_y})
    
    return coords

def main():
    print("🔍 COMPREHENSIVE ASPECT RATIO AND COORDINATE ANALYSIS")
    print("=" * 65)
    
    # Analyze aspect ratio distortion
    analyze_aspect_ratio_distortion()
    
    # Test moon bias specifically
    test_moon_bias()
    
    print("\n📊 SUMMARY OF FINDINGS:")
    print("1. Aspect ratio distortion from 600x400 → 28x28 severely affects recognition")
    print("2. Horizontal shapes get vertically compressed")
    print("3. Vertical shapes get horizontally stretched") 
    print("4. Moon predictions may be biased due to crescent shape distortion")
    print("5. Low confidence likely due to shape distortion making features unrecognizable")
    
    print("\n🔧 RECOMMENDED SOLUTIONS:")
    print("1. Preserve aspect ratio by padding instead of stretching")
    print("2. Use square canvas coordinates from frontend")
    print("3. Improve stroke separation to maintain shape integrity")
    print("4. Consider larger model input size (e.g., 64x64) for better detail preservation")

if __name__ == "__main__":
    main()