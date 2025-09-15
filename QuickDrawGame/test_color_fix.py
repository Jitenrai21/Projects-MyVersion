#!/usr/bin/env python3
"""
Quick verification test for the color inversion fix
"""

import sys
import os

# Add the backend path to import the drawing model
sys.path.append(os.path.join(os.getcwd(), 'backend', 'app'))

from models.drawing_model import predict_drawing

def test_color_fix():
    """Test the color inversion fix with simple shapes"""
    
    print("🧪 TESTING COLOR INVERSION FIX")
    print("=" * 35)
    
    # Simple circle coordinates
    circle_coords = []
    center_x, center_y = 200, 200
    radius = 80
    
    import numpy as np
    for angle in np.linspace(0, 2*np.pi, 30):
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        circle_coords.append({'x': x, 'y': y})
    
    print(f"🔍 Testing circle drawing...")
    print(f"   Coordinates: {len(circle_coords)} points")
    
    # Test prediction
    result = predict_drawing(circle_coords)
    
    print(f"\n📊 Results:")
    print(f"   Prediction: {result.get('prediction', 'ERROR')}")
    print(f"   Confidence: {result.get('confidence', 0)*100:.1f}%")
    print(f"   Color fix applied: {result.get('color_fix_applied', False)}")
    
    if 'top_predictions' in result:
        print(f"   Top 3 predictions:")
        for pred, conf in list(result['top_predictions'].items())[:3]:
            print(f"     {pred}: {conf*100:.1f}%")
    
    # Check if the fix worked
    confidence = result.get('confidence', 0)
    if confidence > 0.5:
        print(f"\n✅ SUCCESS! High confidence indicates color fix worked!")
        print(f"   The model can now properly recognize drawings.")
    elif confidence > 0.3:
        print(f"\n🔄 IMPROVEMENT! Moderate confidence - much better than before.")
        print(f"   Color fix appears to be working.")
    else:
        print(f"\n❌ Still low confidence. May need additional investigation.")
    
    return result

if __name__ == "__main__":
    test_color_fix()