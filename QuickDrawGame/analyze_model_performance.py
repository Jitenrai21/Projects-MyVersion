#!/usr/bin/env python3
"""
Model Performance Analysis Script
This script analyzes the improved model performance and identifies root causes
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter
import pickle
import os

def load_and_analyze_improved_model():
    """Load and analyze the improved model"""
    
    print("🔍 IMPROVED MODEL ANALYSIS")
    print("=" * 40)
    
    model_path = "model_training/model_trad/QuickDraw_improved_final.keras"
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        
        print(f"\n📊 Model Architecture:")
        model.summary()
        
        print(f"\n📋 Model Details:")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Total parameters: {model.count_params():,}")
        print(f"   Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def analyze_training_data_sample():
    """Analyze a sample from training data to understand expected input format"""
    
    print(f"\n🔍 TRAINING DATA SAMPLE ANALYSIS")
    print("=" * 45)
    
    try:
        # Load training data
        with open("features_onTrad", "rb") as f:
            features = pickle.load(f)
        with open("labels_onTrad", "rb") as f:
            labels = pickle.load(f)
        
        print(f"✅ Training data loaded: {features.shape}")
        
        # Analyze sample
        sample_idx = 0
        sample_image = features[sample_idx].reshape(28, 28)
        sample_label = int(labels[sample_idx])
        
        print(f"\n📊 Sample Analysis:")
        print(f"   Sample shape: {sample_image.shape}")
        print(f"   Value range: {sample_image.min():.3f} to {sample_image.max():.3f}")
        print(f"   Mean: {sample_image.mean():.3f}")
        print(f"   Std: {sample_image.std():.3f}")
        print(f"   Non-zero pixels: {np.sum(sample_image > 0.1)} / {sample_image.size}")
        print(f"   Label: {sample_label}")
        
        # Visualize training sample
        plt.figure(figsize=(4, 4))
        plt.imshow(sample_image, cmap='gray', vmin=0, vmax=1)
        plt.title(f'Training Sample (Label: {sample_label})')
        plt.axis('off')
        plt.savefig('training_sample_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return sample_image, sample_label
        
    except Exception as e:
        print(f"❌ Error analyzing training data: {e}")
        return None, None

def create_synthetic_inference_data():
    """Create synthetic data that matches the inference pipeline"""
    
    print(f"\n🎨 CREATING SYNTHETIC INFERENCE DATA")
    print("=" * 45)
    
    # Simulate drawing coordinates for different shapes
    synthetic_drawings = {
        'circle': generate_circle_coordinates(),
        'square': generate_square_coordinates(),
        'star': generate_star_coordinates(),
        'simple_line': generate_line_coordinates()
    }
    
    return synthetic_drawings

def generate_circle_coordinates():
    """Generate coordinates for a circle drawing"""
    center_x, center_y = 200, 200  # Center of 400x400 canvas
    radius = 80
    coordinates = []
    
    for angle in np.linspace(0, 2*np.pi, 50):
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        coordinates.append({'x': x, 'y': y})
    
    return coordinates

def generate_square_coordinates():
    """Generate coordinates for a square drawing"""
    size = 120
    center_x, center_y = 200, 200
    
    coordinates = []
    # Top line
    for x in np.linspace(center_x - size//2, center_x + size//2, 20):
        coordinates.append({'x': x, 'y': center_y - size//2})
    
    # Right line
    for y in np.linspace(center_y - size//2, center_y + size//2, 20):
        coordinates.append({'x': center_x + size//2, 'y': y})
    
    # Bottom line
    for x in np.linspace(center_x + size//2, center_x - size//2, 20):
        coordinates.append({'x': x, 'y': center_y + size//2})
    
    # Left line
    for y in np.linspace(center_y + size//2, center_y - size//2, 20):
        coordinates.append({'x': center_x - size//2, 'y': y})
    
    return coordinates

def generate_star_coordinates():
    """Generate coordinates for a star drawing"""
    center_x, center_y = 200, 200
    outer_radius = 80
    inner_radius = 40
    
    coordinates = []
    for i in range(10):  # 5 points, each with outer and inner
        angle = i * np.pi / 5
        if i % 2 == 0:
            radius = outer_radius
        else:
            radius = inner_radius
        
        x = center_x + radius * np.cos(angle - np.pi/2)
        y = center_y + radius * np.sin(angle - np.pi/2)
        coordinates.append({'x': x, 'y': y})
    
    # Close the star
    coordinates.append(coordinates[0])
    
    return coordinates

def generate_line_coordinates():
    """Generate coordinates for a simple diagonal line"""
    coordinates = []
    for i in range(50):
        x = 100 + i * 4  # From 100 to 300
        y = 100 + i * 4  # Diagonal line
        coordinates.append({'x': x, 'y': y})
    
    return coordinates

def adapt_image_for_training_match(img_array):
    """Apply domain adaptation (copied from drawing_model.py)"""
    
    # Apply slight gaussian blur to simulate training data style
    img_blurred = gaussian_filter(img_array, sigma=0.7)
    
    # Increase contrast slightly to make strokes more prominent
    img_contrasted = np.clip(img_blurred * 1.3, 0, 1)
    
    # Apply slight noise to simulate training data artifacts
    noise = np.random.normal(0, 0.02, img_contrasted.shape)
    img_noisy = np.clip(img_contrasted + noise, 0, 1)
    
    return img_noisy

def preprocess_synthetic_drawing(drawing_coordinates):
    """Preprocess synthetic drawing using the same pipeline as inference"""
    
    canvas_size = (400, 400)
    target_size = (64, 64)
    
    try:
        # Create image on 400x400 canvas
        img = Image.new('L', canvas_size, color=255)  # White background
        draw = ImageDraw.Draw(img)
        
        # Draw the coordinates
        line_width = max(3, min(6, int(min(canvas_size) / 80)))
        
        if len(drawing_coordinates) > 1:
            for i in range(len(drawing_coordinates) - 1):
                x1, y1 = int(drawing_coordinates[i]['x']), int(drawing_coordinates[i]['y'])
                x2, y2 = int(drawing_coordinates[i + 1]['x']), int(drawing_coordinates[i + 1]['y'])
                draw.line([(x1, y1), (x2, y2)], fill=0, width=line_width)
        
        # Apply blur for anti-aliasing
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Resize to target size
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        
        # Apply domain adaptation
        img_array = adapt_image_for_training_match(img_array)
        
        # Reshape for model input
        img_array = img_array.reshape(1, target_size[0], target_size[1], 1)
        
        return img_array, img_resized
        
    except Exception as e:
        print(f"❌ Error preprocessing synthetic drawing: {e}")
        return None, None

def test_model_with_synthetic_data(model, synthetic_drawings):
    """Test the model with synthetic data"""
    
    print(f"\n🧪 TESTING MODEL WITH SYNTHETIC DATA")
    print("=" * 45)
    
    if model is None:
        print("❌ No model available for testing")
        return
    
    CLASS_LABELS = [
        'apple', 'bowtie', 'candle', 'door', 'envelope', 'fish', 'guitar', 'ice cream', 
        'lightning', 'moon', 'mountain', 'star', 'tent', 'toothbrush', 'wristwatch'
    ]
    
    results = {}
    
    for shape_name, coordinates in synthetic_drawings.items():
        print(f"\n🔍 Testing {shape_name}:")
        
        # Preprocess the drawing
        processed_img, original_img = preprocess_synthetic_drawing(coordinates)
        
        if processed_img is None:
            print(f"   ❌ Failed to preprocess {shape_name}")
            continue
        
        # Test with current model input size (should be 28x28)
        input_shape = model.input_shape[1:3]
        print(f"   Model expects: {input_shape}")
        print(f"   Processed shape: {processed_img.shape[1:3]}")
        
        # Resize if needed
        if processed_img.shape[1:3] != input_shape:
            from PIL import Image
            img_pil = Image.fromarray((processed_img[0, :, :, 0] * 255).astype(np.uint8))
            img_resized = img_pil.resize(input_shape, Image.Resampling.LANCZOS)
            processed_img = np.array(img_resized, dtype=np.float32) / 255.0
            processed_img = processed_img.reshape(1, input_shape[0], input_shape[1], 1)
            print(f"   Resized to: {processed_img.shape}")
        
        # Make prediction
        try:
            prediction_probs = model.predict(processed_img, verbose=0)
            predicted_class_idx = np.argmax(prediction_probs[0])
            confidence = float(prediction_probs[0][predicted_class_idx])
            predicted_label = CLASS_LABELS[predicted_class_idx]
            
            # Get top 3 predictions
            top_indices = np.argsort(prediction_probs[0])[-3:][::-1]
            top_predictions = [(CLASS_LABELS[idx], float(prediction_probs[0][idx])) for idx in top_indices]
            
            print(f"   Prediction: {predicted_label} ({confidence*100:.1f}%)")
            print(f"   Top 3: {top_predictions}")
            
            results[shape_name] = {
                'prediction': predicted_label,
                'confidence': confidence,
                'top_predictions': top_predictions,
                'all_probs': prediction_probs[0]
            }
            
            # Save visualization
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Original processed image (64x64 or target size)
            axes[0].imshow(original_img, cmap='gray')
            axes[0].set_title(f'{shape_name} (Processed)')
            axes[0].axis('off')
            
            # Model input (28x28)
            axes[1].imshow(processed_img[0, :, :, 0], cmap='gray')
            axes[1].set_title(f'Model Input {processed_img.shape[1:3]}')
            axes[1].axis('off')
            
            # Prediction probabilities
            axes[2].bar(range(len(CLASS_LABELS)), prediction_probs[0])
            axes[2].set_title(f'Predictions: {predicted_label} ({confidence:.2f})')
            axes[2].set_xticks(range(len(CLASS_LABELS)))
            axes[2].set_xticklabels(CLASS_LABELS, rotation=45, ha='right')
            axes[2].set_ylabel('Probability')
            
            plt.tight_layout()
            plt.savefig(f'synthetic_test_{shape_name}.png', dpi=150, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"   ❌ Prediction error: {e}")
            results[shape_name] = {'error': str(e)}
    
    return results

def analyze_prediction_patterns(results):
    """Analyze patterns in the prediction results"""
    
    print(f"\n📊 PREDICTION PATTERN ANALYSIS")
    print("=" * 40)
    
    # Check for bias patterns
    all_predictions = [r['prediction'] for r in results.values() if 'prediction' in r]
    prediction_counts = {}
    
    for pred in all_predictions:
        prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
    
    print(f"🔍 Prediction Distribution:")
    for pred, count in prediction_counts.items():
        print(f"   {pred}: {count} times")
    
    # Check confidence levels
    all_confidences = [r['confidence'] for r in results.values() if 'confidence' in r]
    if all_confidences:
        avg_confidence = np.mean(all_confidences)
        print(f"\n📊 Confidence Analysis:")
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   Min confidence: {min(all_confidences):.3f}")
        print(f"   Max confidence: {max(all_confidences):.3f}")
        
        if avg_confidence < 0.3:
            print(f"   ⚠️ Very low average confidence - model is uncertain")
        elif avg_confidence < 0.6:
            print(f"   ⚠️ Low average confidence - model needs improvement")
        else:
            print(f"   ✅ Good confidence levels")

def main():
    print("🔍 COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
    print("=" * 55)
    print("Analyzing improved model performance with synthetic inference data\n")
    
    # Step 1: Load and analyze the improved model
    model = load_and_analyze_improved_model()
    
    # Step 2: Analyze training data sample
    training_sample, training_label = analyze_training_data_sample()
    
    # Step 3: Create synthetic inference data
    synthetic_drawings = create_synthetic_inference_data()
    
    # Step 4: Test model with synthetic data
    results = test_model_with_synthetic_data(model, synthetic_drawings)
    
    # Step 5: Analyze prediction patterns
    analyze_prediction_patterns(results)
    
    print(f"\n💡 ROOT CAUSE ANALYSIS:")
    print("=" * 30)
    if model and results:
        # Check for specific issues
        issues_found = []
        
        # Check model input size vs preprocessing
        model_input_size = model.input_shape[1:3]
        if model_input_size != (64, 64):
            issues_found.append(f"Model expects {model_input_size} but preprocessing creates 64x64")
        
        # Check prediction quality
        confidences = [r['confidence'] for r in results.values() if 'confidence' in r]
        if confidences and np.mean(confidences) < 0.5:
            issues_found.append("Low average confidence suggests training or domain issues")
        
        # Check for bias
        predictions = [r['prediction'] for r in results.values() if 'prediction' in r]
        if len(set(predictions)) == 1:
            issues_found.append(f"All predictions are '{predictions[0]}' - severe bias detected")
        
        if issues_found:
            print("🚨 Issues Identified:")
            for i, issue in enumerate(issues_found, 1):
                print(f"   {i}. {issue}")
        else:
            print("✅ No obvious issues detected in basic testing")
            
        print(f"\n🔧 Recommended Next Steps:")
        print("1. Check if model was actually trained properly (20 epochs)")
        print("2. Verify training data quality and preprocessing consistency")
        print("3. Test with more diverse synthetic data")
        print("4. Consider retraining with matched preprocessing pipeline")

if __name__ == "__main__":
    main()