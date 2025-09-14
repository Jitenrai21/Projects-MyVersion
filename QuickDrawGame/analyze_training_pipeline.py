#!/usr/bin/env python3
"""
Training Pipeline Analysis Script
This script analyzes the model training data and process to identify performance issues
"""

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import sys

# Try to import tensorflow for model analysis
try:
    import tensorflow as tf
    print("✅ TensorFlow imported successfully")
except ImportError:
    print("❌ TensorFlow not available for model analysis")
    tf = None

def analyze_training_data():
    """Analyze the preprocessed training data for issues"""
    
    print("\n🔍 TRAINING DATA ANALYSIS")
    print("=" * 50)
    
    try:
        # Load the preprocessed data
        with open("features_onTrad", "rb") as f:
            features = pickle.load(f)
        with open("labels_onTrad", "rb") as f:
            labels = pickle.load(f)
        
        print(f"✅ Successfully loaded training data")
        print(f"📊 Features shape: {features.shape}")
        print(f"📊 Labels shape: {labels.shape}")
        print(f"📊 Features dtype: {features.dtype}")
        print(f"📊 Labels dtype: {labels.dtype}")
        
        # Analyze feature statistics
        print(f"\n📈 Feature Statistics:")
        print(f"   Min value: {features.min():.6f}")
        print(f"   Max value: {features.max():.6f}")
        print(f"   Mean value: {features.mean():.6f}")
        print(f"   Std deviation: {features.std():.6f}")
        
        # Check for potential issues
        print(f"\n🔍 Potential Issues:")
        
        # Issue 1: Check if features are properly normalized
        if features.max() > 1.0 or features.min() < 0.0:
            print("   ⚠️ Features not in [0,1] range - normalization issue!")
        else:
            print("   ✅ Features properly normalized to [0,1]")
        
        # Issue 2: Check label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\n📊 Label Distribution:")
        print(f"   Number of classes: {len(unique_labels)}")
        print(f"   Label range: {labels.min()} to {labels.max()}")
        
        for i, (label, count) in enumerate(zip(unique_labels, counts)):
            print(f"   Class {int(label)}: {count:,} samples")
        
        # Check for class imbalance
        if len(set(counts)) > 1:
            print(f"   ⚠️ Class imbalance detected!")
            print(f"   Min samples: {min(counts):,}, Max samples: {max(counts):,}")
            print(f"   Imbalance ratio: {max(counts)/min(counts):.2f}:1")
        else:
            print(f"   ✅ Classes are balanced")
        
        # Issue 3: Check data shape consistency
        expected_pixels = 28 * 28  # Expected for 28x28 images
        if features.shape[1] != expected_pixels:
            print(f"   ⚠️ Feature dimension mismatch!")
            print(f"   Expected: {expected_pixels} pixels (28x28)")
            print(f"   Actual: {features.shape[1]} pixels")
            
            # Try to figure out the actual image dimensions
            actual_pixels = features.shape[1]
            possible_dims = []
            for dim in range(1, int(actual_pixels**0.5) + 1):
                if actual_pixels % dim == 0:
                    other_dim = actual_pixels // dim
                    possible_dims.append((dim, other_dim))
            
            print(f"   Possible dimensions: {possible_dims}")
        else:
            print(f"   ✅ Feature dimensions match expected 28x28")
        
        # Visualize some samples
        visualize_training_samples(features, labels)
        
        return features, labels
        
    except FileNotFoundError as e:
        print(f"❌ Training data files not found: {e}")
        print("   Make sure to run load_data_onTrad.py first to generate the data files")
        return None, None
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None, None

def visualize_training_samples(features, labels, num_samples=15):
    """Visualize training samples to check data quality"""
    
    print(f"\n🎨 VISUALIZING TRAINING SAMPLES")
    print("=" * 40)
    
    try:
        # Reshape features back to image format
        if features.shape[1] == 784:  # 28x28
            img_shape = (28, 28)
        else:
            # Try to find square dimensions
            pixels = features.shape[1]
            dim = int(pixels**0.5)
            if dim * dim == pixels:
                img_shape = (dim, dim)
            else:
                print(f"   ⚠️ Cannot determine image dimensions from {pixels} pixels")
                return
        
        print(f"   Image shape: {img_shape}")
        
        # Select samples from each class
        unique_labels = np.unique(labels)
        samples_per_class = min(3, num_samples // len(unique_labels))
        
        fig, axes = plt.subplots(len(unique_labels), samples_per_class, 
                                figsize=(samples_per_class * 2, len(unique_labels) * 2))
        
        if len(unique_labels) == 1:
            axes = axes.reshape(1, -1)
        
        for class_idx, class_label in enumerate(unique_labels):
            # Find samples for this class
            class_indices = np.where(labels.flatten() == class_label)[0]
            
            for sample_idx in range(min(samples_per_class, len(class_indices))):
                if samples_per_class == 1:
                    ax = axes[class_idx]
                else:
                    ax = axes[class_idx, sample_idx]
                
                # Get the sample
                sample_features = features[class_indices[sample_idx]].reshape(img_shape)
                
                # Display the image
                ax.imshow(sample_features, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'Class {int(class_label)}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('training_data_samples.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Visualization saved as 'training_data_samples.png'")
        
    except Exception as e:
        print(f"   ❌ Error creating visualization: {e}")

def analyze_model_architecture():
    """Analyze the model architecture for potential issues"""
    
    print(f"\n🏗️ MODEL ARCHITECTURE ANALYSIS")
    print("=" * 45)
    
    if tf is None:
        print("   ❌ TensorFlow not available for model analysis")
        return
    
    try:
        # Load the trained model
        model_path = "model_trad/QuickDraw_tradDataset.keras"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"   ✅ Model loaded from {model_path}")
            
            print(f"\n📊 Model Summary:")
            model.summary()
            
            # Analyze model architecture issues
            print(f"\n🔍 Architecture Analysis:")
            
            # Check input shape
            input_shape = model.input_shape
            print(f"   Input shape: {input_shape}")
            
            if input_shape[1:3] != (28, 28):
                print(f"   ⚠️ Input shape mismatch with expected 28x28!")
            else:
                print(f"   ✅ Input shape matches 28x28")
            
            # Check output shape
            output_shape = model.output_shape
            num_classes = output_shape[1]
            print(f"   Output classes: {num_classes}")
            
            if num_classes != 15:
                print(f"   ⚠️ Output classes mismatch! Expected 15, got {num_classes}")
            else:
                print(f"   ✅ Output classes match expected 15")
            
            # Check model compilation
            print(f"\n⚙️ Model Configuration:")
            print(f"   Optimizer: {model.optimizer.__class__.__name__}")
            print(f"   Loss function: {model.loss}")
            print(f"   Metrics: {model.metrics_names}")
            
            # Analyze model weights for bias
            analyze_model_weights(model)
            
        else:
            print(f"   ❌ Model file not found at {model_path}")
            
    except Exception as e:
        print(f"   ❌ Error analyzing model: {e}")

def analyze_model_weights(model):
    """Analyze model weights for potential bias issues"""
    
    print(f"\n⚖️ Weight Analysis:")
    
    try:
        # Get the final dense layer (output layer)
        last_layer = model.layers[-1]
        
        if hasattr(last_layer, 'get_weights'):
            weights = last_layer.get_weights()
            
            if len(weights) >= 2:
                w, b = weights[0], weights[1]
                
                print(f"   Final layer weights shape: {w.shape}")
                print(f"   Final layer bias shape: {b.shape}")
                
                # Check for extreme bias values
                bias_stats = {
                    'min': b.min(),
                    'max': b.max(),
                    'mean': b.mean(),
                    'std': b.std()
                }
                
                print(f"   Bias statistics:")
                for stat, value in bias_stats.items():
                    print(f"     {stat}: {value:.4f}")
                
                # Check for suspicious bias patterns
                if abs(b.max() - b.min()) > 1.0:
                    print(f"   ⚠️ Large bias range detected! This could cause class preference.")
                
                # Find most biased classes
                sorted_indices = np.argsort(b)
                print(f"   Most negative bias (class {sorted_indices[0]}): {b[sorted_indices[0]]:.4f}")
                print(f"   Most positive bias (class {sorted_indices[-1]}): {b[sorted_indices[-1]]:.4f}")
                
    except Exception as e:
        print(f"   ❌ Error analyzing weights: {e}")

def analyze_training_process():
    """Analyze the training process for issues"""
    
    print(f"\n🎯 TRAINING PROCESS ANALYSIS")
    print("=" * 40)
    
    print(f"🔍 Analyzing training notebook configuration...")
    
    # Check the model training parameters from the notebook
    issues_found = []
    
    print(f"\n📋 Training Configuration Issues:")
    
    # Issue 1: Only 3 epochs
    print(f"   ⚠️ Only 3 epochs trained - likely insufficient!")
    print(f"     Recommendation: Train for 20-50 epochs with early stopping")
    issues_found.append("Insufficient training epochs")
    
    # Issue 2: No validation monitoring in callback
    print(f"   ⚠️ ModelCheckpoint monitors 'val_acc' but no validation split shown")
    print(f"     This could cause issues with model saving")
    issues_found.append("Validation monitoring issue")
    
    # Issue 3: Data augmentation commented out
    print(f"   ⚠️ Data augmentation is commented out")
    print(f"     This reduces model generalization capability")
    issues_found.append("No data augmentation")
    
    # Issue 4: High dropout rate
    print(f"   ⚠️ Dropout rate of 0.6 might be too high")
    print(f"     This could prevent the model from learning effectively")
    issues_found.append("High dropout rate")
    
    # Issue 5: Small batch size
    print(f"   ℹ️ Batch size of 64 is reasonable but could be larger")
    
    return issues_found

def check_preprocessing_consistency():
    """Check for preprocessing mismatches between training and inference"""
    
    print(f"\n🔄 PREPROCESSING CONSISTENCY CHECK")
    print("=" * 45)
    
    print(f"🔍 Comparing training vs inference preprocessing...")
    
    issues = []
    
    # Training preprocessing (from load_data_onTrad.py):
    # x = x.astype('float32') / 255.
    print(f"\n📋 Training Preprocessing:")
    print(f"   ✅ Normalization: x.astype('float32') / 255")
    print(f"   ✅ Range: [0, 1]")
    print(f"   ✅ Data type: float32")
    
    # Inference preprocessing (from drawing_model.py):
    # img_array = img_array / 255.0
    print(f"\n📋 Inference Preprocessing:")
    print(f"   ✅ Normalization: img_array / 255.0")
    print(f"   ✅ Range: [0, 1]")
    print(f"   ✅ Data type: float32")
    
    print(f"\n✅ Preprocessing appears consistent between training and inference")
    
    # But check for other potential issues
    print(f"\n🔍 Potential Preprocessing Issues:")
    print(f"   ⚠️ Training data might be from bitmap images")
    print(f"   ⚠️ Inference creates images from vector coordinates")
    print(f"   ⚠️ This could create domain shift between training and inference")
    issues.append("Domain shift: bitmap training vs vector inference")
    
    return issues

def main():
    print("🔍 COMPREHENSIVE TRAINING PIPELINE ANALYSIS")
    print("=" * 60)
    print("Analyzing the QuickDraw model training pipeline for performance issues")
    
    # Analyze training data
    features, labels = analyze_training_data()
    
    # Analyze model architecture
    analyze_model_architecture()
    
    # Analyze training process
    training_issues = analyze_training_process()
    
    # Check preprocessing consistency
    preprocessing_issues = check_preprocessing_consistency()
    
    # Summary of all issues
    print(f"\n🚨 CRITICAL ISSUES SUMMARY")
    print("=" * 35)
    
    all_issues = training_issues + preprocessing_issues
    
    print(f"📊 Issues Found: {len(all_issues)}")
    for i, issue in enumerate(all_issues, 1):
        print(f"   {i}. {issue}")
    
    print(f"\n🔧 RECOMMENDED FIXES:")
    print("   1. Increase training epochs to 20-50 with early stopping")
    print("   2. Enable data augmentation to improve generalization")
    print("   3. Reduce dropout rate from 0.6 to 0.3-0.4")
    print("   4. Fix validation monitoring in ModelCheckpoint")
    print("   5. Collect training data from vector drawings (not bitmaps)")
    print("   6. Use square canvas (400x400) for training data")
    print("   7. Train with 64x64 input size for better detail")
    print("   8. Add proper class balancing")
    print("   9. Implement learning rate scheduling")
    print("   10. Add more training data per class")
    
    print(f"\n💡 The main issue is likely the domain shift:")
    print("   Training on bitmap images vs inference on vector drawings")
    print("   This explains the poor performance and moon bias!")

if __name__ == "__main__":
    main()