#!/usr/bin/env python3
"""
Deep analysis of the remaining moon bias issue
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

def analyze_training_labels():
    """Analyze the training data label distribution"""
    
    print("🔍 TRAINING DATA LABEL ANALYSIS")
    print("=" * 40)
    
    try:
        with open("labels_onTrad", "rb") as f:
            labels = pickle.load(f)
        
        labels_array = np.array(labels).flatten()
        
        CLASS_LABELS = [
            'apple', 'bowtie', 'candle', 'door', 'envelope', 'fish', 'guitar', 'ice cream',
            'lightning', 'moon', 'mountain', 'star', 'tent', 'toothbrush', 'wristwatch'
        ]
        
        print(f"📊 Label distribution:")
        unique_labels, counts = np.unique(labels_array, return_counts=True)
        
        for label_idx, count in zip(unique_labels, counts):
            class_name = CLASS_LABELS[int(label_idx)]
            print(f"   {int(label_idx):2d} - {class_name:12s}: {count:,} samples")
        
        # Check if moon is overrepresented
        moon_idx = 9  # moon is at index 9
        moon_count = counts[moon_idx] if moon_idx < len(counts) else 0
        total_samples = sum(counts)
        moon_percentage = (moon_count / total_samples) * 100
        
        print(f"\n🌙 Moon class analysis:")
        print(f"   Moon samples: {moon_count:,}")
        print(f"   Total samples: {total_samples:,}")
        print(f"   Moon percentage: {moon_percentage:.1f}%")
        
        if moon_percentage > 8:  # Should be ~6.67% for balanced classes
            print(f"   ⚠️ Moon class appears overrepresented!")
        else:
            print(f"   ✅ Moon class properly balanced")
            
    except Exception as e:
        print(f"❌ Error analyzing labels: {e}")

def check_model_weights():
    """Check if the model has bias toward moon class"""
    
    print(f"\n🔍 MODEL BIAS ANALYSIS")
    print("=" * 30)
    
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model("model_training/model_trad/QuickDraw_improved_final.keras")
        
        # Get the output layer weights
        output_layer = model.layers[-1]
        weights, biases = output_layer.get_weights()
        
        CLASS_LABELS = [
            'apple', 'bowtie', 'candle', 'door', 'envelope', 'fish', 'guitar', 'ice cream',
            'lightning', 'moon', 'mountain', 'star', 'tent', 'toothbrush', 'wristwatch'
        ]
        
        print(f"📊 Output layer biases:")
        for i, (class_name, bias) in enumerate(zip(CLASS_LABELS, biases)):
            print(f"   {i:2d} - {class_name:12s}: {bias:8.4f}")
        
        # Check if moon has higher bias
        moon_idx = 9
        moon_bias = biases[moon_idx]
        avg_bias = np.mean(biases)
        
        print(f"\n🌙 Moon bias analysis:")
        print(f"   Moon bias: {moon_bias:.4f}")
        print(f"   Average bias: {avg_bias:.4f}")
        print(f"   Difference: {moon_bias - avg_bias:+.4f}")
        
        if moon_bias > avg_bias + 0.5:
            print(f"   ⚠️ Moon has significantly higher bias!")
        else:
            print(f"   ✅ Moon bias appears normal")
            
        # Check which class has highest bias
        max_bias_idx = np.argmax(biases)
        max_bias_class = CLASS_LABELS[max_bias_idx]
        print(f"\n   Highest bias class: {max_bias_class} ({biases[max_bias_idx]:.4f})")
        
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")

def test_different_shapes():
    """Test with very different shapes to see if moon bias persists"""
    
    print(f"\n🎨 TESTING DIVERSE SHAPES")
    print("=" * 30)
    
    import sys
    import os
    sys.path.append(os.path.join(os.getcwd(), 'backend', 'app'))
    
    try:
        from models.drawing_model import predict_drawing
        
        # Test 1: Simple apple (circular)
        apple_coords = []
        center_x, center_y = 200, 200
        radius = 60
        import numpy as np
        for angle in np.linspace(0, 2*np.pi, 20):
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            apple_coords.append({'x': x, 'y': y})
        
        print(f"🍎 Testing apple shape:")
        result = predict_drawing(apple_coords)
        print(f"   Prediction: {result.get('prediction')} ({result.get('confidence', 0)*100:.1f}%)")
        
        # Test 2: Rectangle (door)
        door_coords = []
        for x in range(150, 251, 5):  # Top
            door_coords.append({'x': x, 'y': 120})
        for y in range(120, 281, 5):  # Right
            door_coords.append({'x': 250, 'y': y})
        for x in range(250, 149, -5):  # Bottom
            door_coords.append({'x': x, 'y': 280})
        for y in range(280, 119, -5):  # Left
            door_coords.append({'x': 150, 'y': y})
        
        print(f"🚪 Testing door shape:")
        result = predict_drawing(door_coords)
        print(f"   Prediction: {result.get('prediction')} ({result.get('confidence', 0)*100:.1f}%)")
        
        # Test 3: Star shape
        star_coords = []
        center_x, center_y = 200, 200
        outer_radius = 80
        inner_radius = 40
        
        for i in range(10):
            angle = i * np.pi / 5
            radius = outer_radius if i % 2 == 0 else inner_radius
            x = center_x + radius * np.cos(angle - np.pi/2)
            y = center_y + radius * np.sin(angle - np.pi/2)
            star_coords.append({'x': x, 'y': y})
        
        print(f"⭐ Testing star shape:")
        result = predict_drawing(star_coords)
        print(f"   Prediction: {result.get('prediction')} ({result.get('confidence', 0)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error testing shapes: {e}")

def main():
    print("🔍 DEEP ANALYSIS: MOON BIAS INVESTIGATION")
    print("=" * 50)
    
    # Analyze training data
    analyze_training_labels()
    
    # Check model biases
    check_model_weights()
    
    # Test different shapes
    test_different_shapes()
    
    print(f"\n💡 ANALYSIS SUMMARY:")
    print("=" * 25)
    print("1. Color fix improved confidence from ~17% to ~31%")
    print("2. But moon bias still persists - investigating why")
    print("3. Need to check training data balance and model weights")

if __name__ == "__main__":
    main()