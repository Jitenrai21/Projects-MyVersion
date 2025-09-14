#!/usr/bin/env python3
"""
Data Source Analysis Script
This script examines the original NPY data to understand the domain shift issue
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_original_data_source():
    """Analyze the original NPY files to understand data format"""
    
    print("🔍 ORIGINAL DATA SOURCE ANALYSIS")
    print("=" * 45)
    
    data_path = r"D:\QuickDrawDataset(npyfiles)-Trad"
    
    if not os.path.exists(data_path):
        print(f"❌ Original data path not found: {data_path}")
        print("   Cannot analyze original data source")
        return
    
    print(f"✅ Found original data directory: {data_path}")
    
    # List files in the directory
    files = os.listdir(data_path)
    print(f"📁 Found {len(files)} data files:")
    
    class_names = []
    for i, file in enumerate(files[:5]):  # Show first 5 files
        print(f"   {i}: {file}")
        class_names.append(file.replace('.npy', ''))
    
    if len(files) > 5:
        print(f"   ... and {len(files) - 5} more files")
    
    print(f"\n🔍 Analyzing first data file: {files[0]}")
    
    try:
        # Load first file to understand format
        first_file_path = os.path.join(data_path, files[0])
        data = np.load(first_file_path)
        
        print(f"📊 Data shape: {data.shape}")
        print(f"📊 Data type: {data.dtype}")
        print(f"📊 Data range: {data.min():.2f} to {data.max():.2f}")
        print(f"📊 Data mean: {data.mean():.4f}")
        print(f"📊 Data std: {data.std():.4f}")
        
        # Determine if this is bitmap data or vector data
        if len(data.shape) == 2:
            if data.shape[1] == 784:  # 28x28 flattened
                print(f"✅ Data appears to be 28x28 bitmap images (flattened)")
                img_shape = (28, 28)
            elif data.shape[1] == 1024:  # 32x32 flattened
                print(f"✅ Data appears to be 32x32 bitmap images (flattened)")
                img_shape = (32, 32)
            else:
                print(f"❓ Unknown data format - {data.shape[1]} features per sample")
                # Try to find square dimensions
                pixels = data.shape[1]
                dim = int(pixels**0.5)
                if dim * dim == pixels:
                    img_shape = (dim, dim)
                    print(f"   Likely {dim}x{dim} images")
                else:
                    print(f"   Not square image data")
                    return
        elif len(data.shape) == 3:
            print(f"✅ Data appears to be 2D images with shape {data.shape[1:3]}")
            img_shape = data.shape[1:3]
        else:
            print(f"❓ Unknown data format with {len(data.shape)} dimensions")
            return
        
        # Visualize some samples
        visualize_original_samples(data, files[0], img_shape)
        
        # Check if data looks like QuickDraw strokes vs bitmaps
        analyze_data_nature(data, img_shape)
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")

def visualize_original_samples(data, filename, img_shape, num_samples=6):
    """Visualize samples from the original data"""
    
    print(f"\n🎨 VISUALIZING ORIGINAL DATA SAMPLES")
    print("=" * 40)
    
    try:
        # Select random samples
        sample_indices = np.random.choice(len(data), num_samples, replace=False)
        
        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        axes = axes.flatten()
        
        for i, sample_idx in enumerate(sample_indices):
            sample = data[sample_idx]
            
            # Reshape to image format
            if len(sample.shape) == 1:
                sample_img = sample.reshape(img_shape)
            else:
                sample_img = sample
            
            axes[i].imshow(sample_img, cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(f'Sample {sample_idx}')
            axes[i].axis('off')
        
        plt.suptitle(f'Original Data Samples from {filename}')
        plt.tight_layout()
        plt.savefig('original_data_samples.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Visualization saved as 'original_data_samples.png'")
        
    except Exception as e:
        print(f"   ❌ Error creating visualization: {e}")

def analyze_data_nature(data, img_shape):
    """Analyze whether data is bitmap or vector-based"""
    
    print(f"\n🔬 DATA NATURE ANALYSIS")
    print("=" * 30)
    
    # Take a sample for analysis
    sample = data[0].reshape(img_shape)
    
    # Check for continuous grayscale values (bitmap) vs binary values (vector)
    unique_values = np.unique(sample)
    num_unique = len(unique_values)
    
    print(f"📊 Unique pixel values in first sample: {num_unique}")
    
    if num_unique <= 10:
        print(f"   Values: {unique_values}")
        
    if num_unique == 2 and set(unique_values) == {0, 1}:
        print(f"   ✅ Data appears to be BINARY (vector-based drawings)")
        print(f"   This suggests data is from stroke coordinates converted to binary images")
        data_type = "vector"
    elif num_unique <= 10:
        print(f"   ⚠️ Data appears to be DISCRETE with few values")
        print(f"   This could be quantized bitmap or processed vector data")
        data_type = "processed"
    else:
        print(f"   ⚠️ Data appears to be CONTINUOUS grayscale (bitmap)")
        print(f"   This suggests data is from actual image files")
        data_type = "bitmap"
    
    # Check stroke patterns
    analyze_stroke_patterns(sample, data_type)
    
    return data_type

def analyze_stroke_patterns(sample_img, data_type):
    """Analyze stroke patterns in the sample"""
    
    print(f"\n🖊️ STROKE PATTERN ANALYSIS")
    print("=" * 30)
    
    # Calculate stroke thickness and continuity
    non_zero_pixels = np.sum(sample_img > 0.5)
    total_pixels = sample_img.size
    stroke_density = non_zero_pixels / total_pixels
    
    print(f"📊 Stroke density: {stroke_density:.3f} ({non_zero_pixels}/{total_pixels} pixels)")
    
    if stroke_density < 0.05:
        print(f"   ✅ Very sparse strokes - typical of vector drawings")
    elif stroke_density < 0.15:
        print(f"   ✅ Sparse strokes - likely vector-based")
    elif stroke_density < 0.3:
        print(f"   ⚠️ Medium density - could be thick strokes or bitmap")
    else:
        print(f"   ⚠️ High density - likely bitmap or filled shapes")
    
    # Check for continuous lines
    if data_type == "vector":
        print(f"   ✅ Data is vector-based, similar to our inference pipeline")
        print(f"   The domain shift issue is NOT the data format")
    else:
        print(f"   ⚠️ Data might be different format than our inference")
        print(f"   This could explain the domain shift issue")

def compare_with_inference_data():
    """Compare original training data characteristics with our inference pipeline"""
    
    print(f"\n🔄 TRAINING vs INFERENCE COMPARISON")
    print("=" * 45)
    
    print(f"📋 Training Data Characteristics:")
    print(f"   ✅ Source: QuickDraw NPY files (D:\\QuickDrawDataset)")
    print(f"   ✅ Format: 28x28 bitmap images")
    print(f"   ✅ Preprocessing: x.astype('float32') / 255")
    print(f"   ✅ Range: [0, 1]")
    
    print(f"\n📋 Inference Data Characteristics:")
    print(f"   ✅ Source: HTML5 Canvas coordinate drawings")
    print(f"   ✅ Format: Vector coordinates → 28x28 bitmap")
    print(f"   ✅ Preprocessing: img_array / 255.0")
    print(f"   ✅ Range: [0, 1]")
    
    print(f"\n🔍 Potential Domain Shift Sources:")
    print(f"   1. ⚠️ Drawing style differences:")
    print(f"      - Training: Google's QuickDraw data (millions of users)")
    print(f"      - Inference: Your specific drawing style")
    
    print(f"   2. ⚠️ Canvas resolution differences:")
    print(f"      - Training: Original QuickDraw resolution")
    print(f"      - Inference: 400x400 → 28x28 downsampling")
    
    print(f"   3. ⚠️ Stroke thickness differences:")
    print(f"      - Training: QuickDraw stroke thickness")
    print(f"      - Inference: Browser canvas stroke thickness")
    
    print(f"   4. ⚠️ Anti-aliasing differences:")
    print(f"      - Training: QuickDraw processing")
    print(f"      - Inference: PIL ImageDraw anti-aliasing")

def main():
    print("🔍 COMPREHENSIVE DATA SOURCE ANALYSIS")
    print("=" * 50)
    print("Analyzing original training data to understand domain shift")
    
    # Analyze original data source
    analyze_original_data_source()
    
    # Compare with inference pipeline
    compare_with_inference_data()
    
    print(f"\n💡 KEY FINDINGS:")
    print("=" * 20)
    print("1. The training data is likely from Google's QuickDraw dataset")
    print("2. Both training and inference use 28x28 bitmap conversion")
    print("3. The domain shift is likely due to:")
    print("   - Different drawing styles and patterns")
    print("   - Different stroke thickness and rendering")
    print("   - Canvas resolution and downsampling differences")
    print("   - Model trained on only 3 epochs (insufficient)")
    print("   - High dropout rate preventing learning")
    
    print(f"\n🔧 SOLUTION STRATEGY:")
    print("=" * 25)
    print("1. Collect training data using YOUR drawing interface")
    print("2. Use the square 400x400 canvas for data collection")
    print("3. Train with 64x64 input for better detail preservation")
    print("4. Increase training epochs to 20-50")
    print("5. Reduce dropout rate to 0.3-0.4")
    print("6. Enable data augmentation")
    print("7. Use proper validation monitoring")

if __name__ == "__main__":
    main()