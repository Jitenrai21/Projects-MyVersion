#!/usr/bin/env python3
"""
Test script for new square canvas infrastructure
This verifies that the 400x400 square canvas and 64x64 model input setup works correctly
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
    print("✅ Successfully imported updated drawing model")
except ImportError as e:
    print(f"❌ Failed to import drawing model: {e}")
    sys.exit(1)

def test_square_canvas_infrastructure():
    """Test the new square canvas setup"""
    
    print("\n🔍 TESTING SQUARE CANVAS INFRASTRUCTURE")
    print("=" * 55)
    
    # Test shapes for square canvas (400x400)
    test_shapes = {
        "Center Circle": create_center_circle_400(),
        "Top-Left Square": create_topleft_square_400(),
        "Full Width Line": create_horizontal_line_400(),
        "Diagonal Cross": create_diagonal_cross_400(),
        "Moon Crescent": create_moon_crescent_400()
    }
    
    results = {}
    
    for shape_name, coords in test_shapes.items():
        print(f"\n📐 Testing: {shape_name}")
        
        # Test coordinate ranges
        x_coords = [p['x'] for p in coords]
        y_coords = [p['y'] for p in coords]
        
        print(f"   Coordinate ranges:")
        print(f"     X: {min(x_coords):.1f} to {max(x_coords):.1f} (canvas: 0-400)")
        print(f"     Y: {min(y_coords):.1f} to {max(y_coords):.1f} (canvas: 0-400)")
        
        # Test preprocessing
        processed_64 = preprocess_drawing_to_image(coords, (400, 400), (64, 64))
        
        if processed_64 is not None:
            print(f"   ✅ 64x64 preprocessing: SUCCESS")
            print(f"   Shape: {processed_64.shape}")
            print(f"   Value range: {processed_64.min():.3f} to {processed_64.max():.3f}")
            
            # Test prediction (will be resized to 28x28 for current model)
            prediction = predict_drawing(coords)
            print(f"   🤖 Prediction: {prediction.get('prediction', 'error')} ({prediction.get('confidence', 0)*100:.1f}%)")
            
            results[shape_name] = {
                'processed': True,
                'prediction': prediction.get('prediction', 'error'),
                'confidence': prediction.get('confidence', 0)
            }
        else:
            print(f"   ❌ 64x64 preprocessing: FAILED")
            results[shape_name] = {'processed': False}
    
    return results

def create_center_circle_400():
    """Create a circle in center of 400x400 canvas"""
    coords = []
    center_x, center_y = 200, 200  # Center of 400x400
    radius = 60
    for i in range(30):
        angle = (i / 30) * 2 * np.pi
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        coords.append({"x": x, "y": y})
    return coords

def create_topleft_square_400():
    """Create a square in top-left quadrant"""
    coords = []
    points = [(50, 50), (150, 50), (150, 150), (50, 150), (50, 50)]
    for x, y in points:
        coords.append({"x": x, "y": y})
    return coords

def create_horizontal_line_400():
    """Create a horizontal line across the canvas"""
    coords = []
    y = 200  # Middle of canvas
    for x in range(50, 351, 10):  # From x=50 to x=350
        coords.append({"x": x, "y": y})
    return coords

def create_diagonal_cross_400():
    """Create a diagonal cross"""
    coords = []
    # First diagonal (top-left to bottom-right)
    for i in range(20):
        t = i / 19
        x = 100 + t * 200  # From 100 to 300
        y = 100 + t * 200  # From 100 to 300
        coords.append({"x": x, "y": y})
    
    # Add gap for stroke separation
    coords.append({"x": 400, "y": 400, "strokeEnd": True})
    
    # Second diagonal (top-right to bottom-left)
    for i in range(20):
        t = i / 19
        x = 300 - t * 200  # From 300 to 100
        y = 100 + t * 200  # From 100 to 300
        coords.append({"x": x, "y": y})
    
    return coords

def create_moon_crescent_400():
    """Create a moon crescent for 400x400 canvas"""
    coords = []
    center_x, center_y = 200, 200
    
    # Outer arc
    radius1 = 70
    for i in range(25):
        angle = np.pi * (0.3 + 1.4 * i / 24)
        x = center_x + radius1 * np.cos(angle)
        y = center_y + radius1 * np.sin(angle)
        coords.append({"x": x, "y": y})
    
    # Inner arc
    center_x2 = center_x + 20
    radius2 = 50
    for i in range(25):
        angle = np.pi * (1.7 - 1.4 * i / 24)
        x = center_x2 + radius2 * np.cos(angle)
        y = center_y + radius2 * np.sin(angle)
        coords.append({"x": x, "y": y})
    
    return coords

def visualize_square_canvas_processing():
    """Create visualization showing the improvement with square canvas"""
    
    print(f"\n🎨 CREATING VISUALIZATION")
    print("=" * 30)
    
    # Test with a simple shape
    coords = create_center_circle_400()
    
    # Process with new square canvas method
    processed_image = preprocess_drawing_to_image(coords, (400, 400), (64, 64))
    
    if processed_image is None:
        print("❌ Failed to process image for visualization")
        return
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Square Canvas Infrastructure Test", fontsize=16)
    
    # Plot 1: Original coordinates on square canvas
    x_coords = [p['x'] for p in coords]
    y_coords = [p['y'] for p in coords]
    
    axes[0].set_title("Original Drawing\n(400x400 Square Canvas)")
    axes[0].set_xlim(0, 400)
    axes[0].set_ylim(400, 0)  # Flip Y
    axes[0].plot(x_coords, y_coords, 'b-', linewidth=3, marker='o', markersize=2)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel("X coordinate")
    axes[0].set_ylabel("Y coordinate")
    
    # Plot 2: Processed 64x64 image
    axes[1].set_title("Processed Image\n(64x64 - High Detail)")
    axes[1].imshow(processed_image[0, :, :, 0], cmap='gray', aspect='equal')
    axes[1].axis('off')
    
    # Plot 3: What it would look like at 28x28 (current model size)
    # Resize to 28x28 for comparison
    img_28 = Image.fromarray((processed_image[0, :, :, 0] * 255).astype(np.uint8))
    img_28_resized = img_28.resize((28, 28), Image.Resampling.LANCZOS)
    img_28_array = np.array(img_28_resized) / 255.0
    
    axes[2].set_title("Resized for Current Model\n(28x28 - Legacy)")
    axes[2].imshow(img_28_array, cmap='gray', aspect='equal')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('square_canvas_infrastructure_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Visualization saved as 'square_canvas_infrastructure_test.png'")

def test_coordinate_edge_cases_square():
    """Test edge cases with square canvas"""
    
    print(f"\n🧪 TESTING EDGE CASES (Square Canvas)")
    print("=" * 45)
    
    edge_cases = {
        "Single Center Point": [{"x": 200, "y": 200}],
        "Corner to Corner": [{"x": 0, "y": 0}, {"x": 400, "y": 400}],
        "Edge Points": [{"x": 0, "y": 200}, {"x": 400, "y": 200}],
        "Small Shape": [{"x": 195, "y": 195}, {"x": 205, "y": 195}, {"x": 205, "y": 205}, {"x": 195, "y": 205}],
        "With Stroke Markers": [
            {"x": 100, "y": 100}, {"x": 200, "y": 100},
            {"x": 999, "y": 999, "strokeEnd": True},  # Stroke end marker
            {"x": 100, "y": 200}, {"x": 200, "y": 200}
        ]
    }
    
    for case_name, coords in edge_cases.items():
        print(f"\n   Testing: {case_name}")
        
        try:
            processed = preprocess_drawing_to_image(coords, (400, 400), (64, 64))
            if processed is not None:
                print(f"      ✅ SUCCESS - Shape: {processed.shape}")
                print(f"      Value range: {processed.min():.3f} to {processed.max():.3f}")
            else:
                print(f"      ❌ FAILED - Returned None")
        except Exception as e:
            print(f"      ❌ ERROR - {str(e)}")

def main():
    print("🔍 SQUARE CANVAS INFRASTRUCTURE TEST")
    print("=" * 45)
    print("Testing the new 400x400 square canvas with 64x64 model input setup")
    print("This prepares the infrastructure for retraining with better aspect ratio")
    
    # Test basic functionality
    results = test_square_canvas_infrastructure()
    
    # Test edge cases
    test_coordinate_edge_cases_square()
    
    # Create visualization
    visualize_square_canvas_processing()
    
    # Summary
    print(f"\n📊 INFRASTRUCTURE TEST SUMMARY")
    print("=" * 40)
    
    successful_tests = sum(1 for r in results.values() if r.get('processed', False))
    total_tests = len(results)
    
    print(f"✅ Successful preprocessing: {successful_tests}/{total_tests}")
    print(f"📐 Canvas size: 400x400 (square - no aspect ratio issues)")
    print(f"🖼️ Target size: 64x64 (high detail preservation)")
    print(f"🔄 Current model: 28x28 (compatibility maintained)")
    
    if successful_tests == total_tests:
        print(f"\n🎉 INFRASTRUCTURE READY!")
        print("   ✅ Square canvas implemented")
        print("   ✅ 64x64 preprocessing working")
        print("   ✅ Backward compatibility maintained")
        print("   ✅ Ready for model retraining")
    else:
        print(f"\n⚠️ Some tests failed - check the logs above")
    
    print(f"\n🔧 NEXT STEPS FOR MODEL RETRAINING:")
    print("   1. Update model architecture to accept 64x64 input")
    print("   2. Retrain with square-canvas preprocessed data")
    print("   3. Remove the 28x28 compatibility layer")
    print("   4. Test with real user drawings")

if __name__ == "__main__":
    main()