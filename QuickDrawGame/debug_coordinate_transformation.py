#!/usr/bin/env python3
"""
Debug script to test coordinate transformation from frontend to backend
This script will help identify issues in the coordinate processing pipeline
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Add the backend directory to the Python path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend', 'app')
sys.path.insert(0, backend_dir)

# Import the drawing model
try:
    from models.drawing_model import preprocess_drawing_to_image, CLASS_LABELS
    print("✅ Successfully imported drawing model")
except ImportError as e:
    print(f"❌ Failed to import drawing model: {e}")
    sys.exit(1)

def create_test_drawings():
    """Create test drawings that should be recognizable"""
    
    # Test 1: Simple circle (apple-like)
    circle_coords = []
    center_x, center_y = 300, 200
    radius = 50
    for i in range(100):
        angle = (i / 100) * 2 * np.pi
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        circle_coords.append({"x": x, "y": y})
    
    # Test 2: Simple star shape
    star_coords = []
    center_x, center_y = 300, 200
    outer_radius = 60
    inner_radius = 30
    for i in range(11):  # 10 points + return to start
        angle = (i / 10) * 2 * np.pi
        if i % 2 == 0:  # Outer points
            x = center_x + outer_radius * np.cos(angle - np.pi/2)
            y = center_y + outer_radius * np.sin(angle - np.pi/2)
        else:  # Inner points
            x = center_x + inner_radius * np.cos(angle - np.pi/2)
            y = center_y + inner_radius * np.sin(angle - np.pi/2)
        star_coords.append({"x": x, "y": y})
    
    # Test 3: Simple envelope (rectangle)
    envelope_coords = []
    # Draw rectangle
    points = [
        (200, 150), (400, 150), (400, 250), (200, 250), (200, 150),
        # Add envelope flap
        (200, 150), (300, 200), (400, 150)
    ]
    for x, y in points:
        envelope_coords.append({"x": x, "y": y})
    
    return {
        "circle_apple": circle_coords,
        "star": star_coords,
        "envelope": envelope_coords
    }

def visualize_transformation(drawing_data, title, canvas_size=(400, 400)):
    """Visualize the transformation process - Updated for square canvas"""
    
    print(f"\n🔍 Testing: {title}")
    print(f"   Original points: {len(drawing_data)}")
    
    if not drawing_data:
        print("   ❌ No drawing data!")
        return
    
    # Get coordinate ranges
    x_coords = [p['x'] for p in drawing_data]
    y_coords = [p['y'] for p in drawing_data]
    
    print(f"   X range: {min(x_coords):.1f} to {max(x_coords):.1f}")
    print(f"   Y range: {min(y_coords):.1f} to {max(y_coords):.1f}")
    
    # Test the preprocessing with new square canvas
    processed_image = preprocess_drawing_to_image(drawing_data, canvas_size, (64, 64))
    
    if processed_image is None:
        print("   ❌ Preprocessing failed!")
        return
    
    print(f"   ✅ Processed image shape: {processed_image.shape}")
    print(f"   Image value range: {processed_image.min():.3f} to {processed_image.max():.3f}")
    
    # Create visualization for square canvas
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Original coordinates on square canvas
    axes[0].set_title(f"Original Drawing: {title}")
    axes[0].set_xlim(0, canvas_size[0])
    axes[0].set_ylim(canvas_size[1], 0)  # Flip Y axis to match canvas
    axes[0].plot(x_coords, y_coords, 'b-', linewidth=2, marker='o', markersize=1)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Recreation of the drawing process
    img_full = Image.new('L', canvas_size, color=255)
    draw = ImageDraw.Draw(img_full)
    
    # Draw lines connecting points
    if len(drawing_data) > 1:
        for i in range(len(drawing_data) - 1):
            x1, y1 = int(drawing_data[i]['x']), int(drawing_data[i]['y'])
            x2, y2 = int(drawing_data[i + 1]['x']), int(drawing_data[i + 1]['y'])
            draw.line([(x1, y1), (x2, y2)], fill=0, width=5)
    
    # Draw individual points
    for point in drawing_data:
        x, y = int(point['x']), int(point['y'])
        draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=0)
    
    axes[1].set_title("Recreated Full Image\n(Square 400x400)")
    axes[1].imshow(img_full, cmap='gray', aspect='equal')
    axes[1].axis('off')
    
    # Plot 3: Final 64x64 image
    axes[2].set_title("Final 64x64 Image\n(No aspect ratio distortion!)")
    axes[2].imshow(processed_image[0, :, :, 0], cmap='gray', aspect='equal')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'debug_square_canvas_{title.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return processed_image

def test_coordinate_edge_cases():
    """Test edge cases in coordinate handling"""
    
    print("\n🧪 Testing Edge Cases:")
    
    # Test 1: Single point
    single_point = [{"x": 300, "y": 200}]
    result = preprocess_drawing_to_image(single_point)
    print(f"   Single point: {'✅ Success' if result is not None else '❌ Failed'}")
    
    # Test 2: Two points (line)
    two_points = [{"x": 200, "y": 200}, {"x": 400, "y": 200}]
    result = preprocess_drawing_to_image(two_points)
    print(f"   Two points: {'✅ Success' if result is not None else '❌ Failed'}")
    
    # Test 3: Points at edges
    edge_points = [
        {"x": 0, "y": 0}, {"x": 599, "y": 0}, 
        {"x": 599, "y": 399}, {"x": 0, "y": 399}, {"x": 0, "y": 0}
    ]
    result = preprocess_drawing_to_image(edge_points)
    print(f"   Edge points: {'✅ Success' if result is not None else '❌ Failed'}")
    
    # Test 4: Decimal coordinates (from frontend)
    decimal_coords = [
        {"x": 300.5, "y": 200.7}, {"x": 301.3, "y": 201.9}, 
        {"x": 302.1, "y": 203.4}
    ]
    result = preprocess_drawing_to_image(decimal_coords)
    print(f"   Decimal coords: {'✅ Success' if result is not None else '❌ Failed'}")
    
    # Test 5: Empty drawing
    empty_drawing = []
    result = preprocess_drawing_to_image(empty_drawing)
    print(f"   Empty drawing: {'✅ Correctly handled' if result is None else '❌ Should be None'}")

def main():
    print("🔍 QuickDraw Coordinate Transformation Debug")
    print("=" * 50)
    
    print(f"📋 Available classes: {CLASS_LABELS}")
    print(f"📊 Total classes: {len(CLASS_LABELS)}")
    
    # Test basic edge cases
    test_coordinate_edge_cases()
    
    # Create and test sample drawings
    test_drawings = create_test_drawings()
    
    for name, coords in test_drawings.items():
        processed = visualize_transformation(coords, name)
        if processed is not None:
            print(f"   🎯 {name}: Processed successfully")
        else:
            print(f"   ❌ {name}: Processing failed")
    
    print("\n📊 Summary:")
    print("   - Check the generated PNG files for visual verification")
    print("   - Look for issues in coordinate scaling and image quality")
    print("   - Verify that drawings remain recognizable after transformation")
    
    print("\n🔧 Potential Issues to Look For:")
    print("   1. Coordinate system mismatch (frontend vs backend)")
    print("   2. Canvas size inconsistencies")
    print("   3. Line thickness affecting small details")
    print("   4. Aspect ratio distortion during resizing")
    print("   5. Class label mismatches (21 vs 15 classes)")

if __name__ == "__main__":
    main()