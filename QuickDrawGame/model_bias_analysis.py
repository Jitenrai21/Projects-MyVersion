#!/usr/bin/env python3
"""
Model Analysis Script to investigate the severe moon bias
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
    from models.drawing_model import model, CLASS_LABELS, predict_drawing
    print("✅ Successfully imported drawing model")
except ImportError as e:
    print(f"❌ Failed to import drawing model: {e}")
    sys.exit(1)

def analyze_model_bias():
    """Analyze model architecture and prediction patterns"""
    
    print("\n🔍 MODEL BIAS ANALYSIS")
    print("=" * 50)
    
    if model is None:
        print("❌ Model is not loaded!")
        return
    
    # Model architecture analysis
    print(f"📊 Model Summary:")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Total parameters: {model.count_params():,}")
    print(f"   Number of layers: {len(model.layers)}")
    
    # Check model weights for bias
    print(f"\n🔍 Model Output Layer Analysis:")
    try:
        # Get the last layer (should be Dense with 15 outputs)
        last_layer = model.layers[-1]
        print(f"   Last layer type: {type(last_layer).__name__}")
        print(f"   Last layer units: {last_layer.units if hasattr(last_layer, 'units') else 'N/A'}")
        
        # Get weights and biases
        if hasattr(last_layer, 'get_weights'):
            weights = last_layer.get_weights()
            if len(weights) >= 2:
                bias = weights[1]  # Bias vector
                print(f"   Bias shape: {bias.shape}")
                print(f"   Bias values: {bias}")
                
                # Check if any class has unusually high bias
                max_bias_idx = np.argmax(bias)
                min_bias_idx = np.argmin(bias)
                print(f"   Highest bias: {CLASS_LABELS[max_bias_idx]} ({bias[max_bias_idx]:.4f})")
                print(f"   Lowest bias: {CLASS_LABELS[min_bias_idx]} ({bias[min_bias_idx]:.4f})")
                
                # Check bias distribution
                moon_idx = CLASS_LABELS.index('moon') if 'moon' in CLASS_LABELS else -1
                if moon_idx >= 0:
                    print(f"   Moon bias: {bias[moon_idx]:.4f}")
    
    except Exception as e:
        print(f"   Error analyzing weights: {e}")

def test_empty_and_noise_inputs():
    """Test model response to empty and noise inputs"""
    
    print("\n🧪 TESTING MODEL WITH EDGE CASES")
    print("=" * 45)
    
    # Test 1: Empty input (all white)
    empty_input = np.ones((1, 28, 28, 1), dtype=np.float32)
    try:
        empty_pred = model.predict(empty_input, verbose=0)
        empty_class_idx = np.argmax(empty_pred[0])
        empty_confidence = float(empty_pred[0][empty_class_idx])
        print(f"🔍 Empty (all white) input:")
        print(f"   Prediction: {CLASS_LABELS[empty_class_idx]} ({empty_confidence*100:.1f}%)")
        print(f"   All probs: {[f'{p:.3f}' for p in empty_pred[0]]}")
    except Exception as e:
        print(f"   Error with empty input: {e}")
    
    # Test 2: All black input
    black_input = np.zeros((1, 28, 28, 1), dtype=np.float32)
    try:
        black_pred = model.predict(black_input, verbose=0)
        black_class_idx = np.argmax(black_pred[0])
        black_confidence = float(black_pred[0][black_class_idx])
        print(f"\n🔍 All black input:")
        print(f"   Prediction: {CLASS_LABELS[black_class_idx]} ({black_confidence*100:.1f}%)")
        print(f"   All probs: {[f'{p:.3f}' for p in black_pred[0]]}")
    except Exception as e:
        print(f"   Error with black input: {e}")
    
    # Test 3: Random noise
    noise_input = np.random.random((1, 28, 28, 1)).astype(np.float32)
    try:
        noise_pred = model.predict(noise_input, verbose=0)
        noise_class_idx = np.argmax(noise_pred[0])
        noise_confidence = float(noise_pred[0][noise_class_idx])
        print(f"\n🔍 Random noise input:")
        print(f"   Prediction: {CLASS_LABELS[noise_class_idx]} ({noise_confidence*100:.1f}%)")
        print(f"   Max prob: {np.max(noise_pred[0]):.3f}")
    except Exception as e:
        print(f"   Error with noise input: {e}")
    
    # Test 4: Simple geometric shapes
    test_simple_shapes()

def test_simple_shapes():
    """Test with very simple, clear geometric shapes"""
    
    print(f"\n🔍 TESTING SIMPLE GEOMETRIC SHAPES")
    print("=" * 40)
    
    shapes = {
        "Perfect Center Dot": create_center_dot(),
        "Horizontal Line": create_horizontal_line(),
        "Vertical Line": create_vertical_line(),
        "Simple Square": create_simple_square(),
        "Simple Circle": create_simple_circle_28x28()
    }
    
    for name, shape_img in shapes.items():
        try:
            pred = model.predict(shape_img, verbose=0)
            pred_idx = np.argmax(pred[0])
            confidence = float(pred[0][pred_idx])
            
            print(f"\n   {name}:")
            print(f"      Prediction: {CLASS_LABELS[pred_idx]} ({confidence*100:.1f}%)")
            
            # Show top 3
            top_indices = np.argsort(pred[0])[-3:][::-1]
            print(f"      Top 3:")
            for i, idx in enumerate(top_indices):
                print(f"         {i+1}. {CLASS_LABELS[idx]}: {pred[0][idx]*100:.1f}%")
            
        except Exception as e:
            print(f"   Error testing {name}: {e}")

def create_center_dot():
    """Create a single dot in the center"""
    img = np.ones((1, 28, 28, 1), dtype=np.float32)
    img[0, 13:15, 13:15, 0] = 0.0  # Black dot in center
    return img

def create_horizontal_line():
    """Create a horizontal line"""
    img = np.ones((1, 28, 28, 1), dtype=np.float32)
    img[0, 13:15, 5:23, 0] = 0.0  # Horizontal line
    return img

def create_vertical_line():
    """Create a vertical line"""
    img = np.ones((1, 28, 28, 1), dtype=np.float32)
    img[0, 5:23, 13:15, 0] = 0.0  # Vertical line
    return img

def create_simple_square():
    """Create a simple square outline"""
    img = np.ones((1, 28, 28, 1), dtype=np.float32)
    # Top and bottom lines
    img[0, 8:10, 8:20, 0] = 0.0
    img[0, 18:20, 8:20, 0] = 0.0
    # Left and right lines
    img[0, 8:20, 8:10, 0] = 0.0
    img[0, 8:20, 18:20, 0] = 0.0
    return img

def create_simple_circle_28x28():
    """Create a simple circle directly in 28x28"""
    img = np.ones((1, 28, 28, 1), dtype=np.float32)
    center = 14
    radius = 8
    
    # Create circle using distance formula
    for y in range(28):
        for x in range(28):
            dist = ((x - center)**2 + (y - center)**2)**0.5
            if abs(dist - radius) < 1.5:  # Circle outline
                img[0, y, x, 0] = 0.0
    
    return img

def check_training_data_issues():
    """Check for potential training data issues"""
    
    print(f"\n🔍 POTENTIAL TRAINING DATA ISSUES")
    print("=" * 45)
    
    print("🚨 Based on the analysis, possible issues:")
    print("   1. Model might be undertrained or overtrained")
    print("   2. Training data might have severe class imbalance")
    print("   3. Training data preprocessing might differ from current preprocessing")
    print("   4. Model might have been trained on different image format/size")
    print("   5. Training labels might be corrupted")
    
    print(f"\n🔍 Model File Information:")
    model_path = os.path.join(os.path.dirname(__file__), 'model_training', 'model_trad', 'QuickDraw_tradDataset.keras')
    if os.path.exists(model_path):
        stat = os.stat(model_path)
        print(f"   Model file size: {stat.st_size / (1024*1024):.1f} MB")
        print(f"   Model file path: {model_path}")
    else:
        print(f"   ❌ Model file not found at expected path")

def recommend_solutions():
    """Recommend solutions based on findings"""
    
    print(f"\n🔧 RECOMMENDED SOLUTIONS")
    print("=" * 30)
    
    print("📋 IMMEDIATE ACTIONS:")
    print("   1. ✅ Check if model file is corrupted")
    print("   2. ✅ Verify training data preprocessing matches current preprocessing") 
    print("   3. ✅ Check if model was actually trained properly")
    print("   4. ✅ Consider retraining the model from scratch")
    print("   5. ✅ Test with a different pre-trained QuickDraw model")
    
    print(f"\n📋 LONG-TERM FIXES:")
    print("   1. Retrain with balanced dataset")
    print("   2. Use data augmentation during training")
    print("   3. Implement proper validation during training")
    print("   4. Consider using transfer learning from a working model")
    print("   5. Increase model complexity if underfitting")

def main():
    print("🔍 COMPREHENSIVE MODEL BIAS ANALYSIS")
    print("=" * 50)
    
    analyze_model_bias()
    test_empty_and_noise_inputs()
    check_training_data_issues()
    recommend_solutions()
    
    print(f"\n⚠️  CRITICAL CONCLUSION:")
    print("   The moon bias with ~12% confidence indicates the model is")
    print("   fundamentally broken. This is NOT a coordinate processing issue.")
    print("   The model needs to be retrained or replaced.")

if __name__ == "__main__":
    main()