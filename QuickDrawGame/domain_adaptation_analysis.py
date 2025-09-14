#!/usr/bin/env python3
"""
Domain Adaptation Analysis Script
This script explores ways to bridge the domain gap between existing NPY data and canvas inference
without requiring manual data collection.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import os
import pickle

def analyze_domain_adaptation_options():
    """Analyze different approaches to handle domain shift without new data collection"""
    
    print("🔄 DOMAIN ADAPTATION STRATEGIES")
    print("=" * 45)
    print("Exploring ways to bridge the gap without manual data collection\n")
    
    # Load existing training data for analysis
    try:
        with open("features_onTrad", "rb") as f:
            features = pickle.load(f)
        print(f"✅ Loaded training data: {features.shape}")
        
        # Analyze current training data characteristics
        sample_img = features[0].reshape(28, 28)
        
        print(f"\n📊 Current Training Data Analysis:")
        print(f"   Resolution: 28x28")
        print(f"   Pixel values: {len(np.unique(sample_img))} unique values")
        print(f"   Value range: {sample_img.min():.3f} to {sample_img.max():.3f}")
        print(f"   Stroke density: {np.mean(sample_img > 0.1):.3f}")
        
        return features
        
    except FileNotFoundError:
        print("❌ Training data not found")
        return None

def strategy_1_preprocess_training_data(features):
    """Strategy 1: Preprocess training data to match canvas inference characteristics"""
    
    print(f"\n🎯 STRATEGY 1: PREPROCESS TRAINING DATA")
    print("=" * 40)
    print("Modify training data to match your canvas inference pipeline\n")
    
    print(f"🔧 Proposed Transformations:")
    
    # 1. Vectorize and re-rasterize
    print(f"   1. VECTORIZATION + RE-RASTERIZATION:")
    print(f"      - Extract stroke paths from bitmap using skeletonization")
    print(f"      - Convert to coordinate sequences")
    print(f"      - Re-render using PIL ImageDraw (same as inference)")
    print(f"      - Result: Training data matches inference rendering")
    
    # 2. Style transfer
    print(f"\n   2. STYLE TRANSFER:")
    print(f"      - Apply morphological operations to thin/thicken strokes")
    print(f"      - Add slight blur to simulate canvas anti-aliasing")
    print(f"      - Adjust stroke connectivity")
    print(f"      - Result: Training style closer to canvas style")
    
    # 3. Resolution upgrade
    print(f"\n   3. RESOLUTION UPGRADE:")
    print(f"      - Upscale 28x28 → 64x64 using interpolation")
    print(f"      - Apply sharpening to maintain crisp edges")
    print(f"      - Retrain model on 64x64 input")
    print(f"      - Result: Better detail preservation")
    
    # Demonstrate vectorization approach
    demonstrate_vectorization(features)
    
    return "preprocess_training"

def strategy_2_adapt_inference_pipeline(features):
    """Strategy 2: Modify inference pipeline to match training data characteristics"""
    
    print(f"\n🎯 STRATEGY 2: ADAPT INFERENCE PIPELINE")
    print("=" * 42)
    print("Modify your canvas inference to match training data characteristics\n")
    
    print(f"🔧 Proposed Modifications:")
    
    # 1. Style matching
    print(f"   1. STYLE MATCHING:")
    print(f"      - Analyze training data stroke characteristics")
    print(f"      - Adjust canvas stroke thickness to match")
    print(f"      - Modify anti-aliasing settings")
    print(f"      - Result: Canvas output closer to training style")
    
    # 2. Post-processing
    print(f"\n   2. POST-PROCESSING:")
    print(f"      - Apply Gaussian blur to canvas images")
    print(f"      - Add noise to simulate training data artifacts")
    print(f"      - Adjust contrast/brightness")
    print(f"      - Result: Canvas images match training distribution")
    
    # 3. Preprocessing alignment
    print(f"\n   3. PREPROCESSING ALIGNMENT:")
    print(f"      - Downscale to 28x28 (current training resolution)")
    print(f"      - Apply same filtering as original training data")
    print(f"      - Match exact pixel value distribution")
    print(f"      - Result: Perfect preprocessing alignment")
    
    return "adapt_inference"

def strategy_3_hybrid_approach():
    """Strategy 3: Hybrid approach combining both strategies"""
    
    print(f"\n🎯 STRATEGY 3: HYBRID APPROACH")
    print("=" * 35)
    print("Combine training data preprocessing with inference adaptation\n")
    
    print(f"🔧 Hybrid Pipeline:")
    print(f"   1. Upgrade training data to 64x64 resolution")
    print(f"   2. Apply style transfer to training data")
    print(f"   3. Slightly modify inference preprocessing")
    print(f"   4. Use data augmentation during training")
    print(f"   5. Train with domain adaptation techniques")
    
    return "hybrid"

def strategy_4_training_improvements():
    """Strategy 4: Focus on training improvements to overcome domain shift"""
    
    print(f"\n🎯 STRATEGY 4: TRAINING IMPROVEMENTS")
    print("=" * 40)
    print("Use better training techniques to handle domain shift\n")
    
    print(f"🔧 Advanced Training Techniques:")
    
    # 1. Data augmentation
    print(f"   1. AGGRESSIVE DATA AUGMENTATION:")
    print(f"      - Random rotations (-15° to +15°)")
    print(f"      - Random scaling (0.8x to 1.2x)")
    print(f"      - Random translations")
    print(f"      - Elastic deformations")
    print(f"      - Gaussian noise addition")
    print(f"      - Result: Model becomes more robust to style differences")
    
    # 2. Regularization
    print(f"\n   2. DOMAIN ROBUST REGULARIZATION:")
    print(f"      - Reduce dropout from 0.6 to 0.3")
    print(f"      - Add batch normalization")
    print(f"      - Use label smoothing")
    print(f"      - Implement mixup augmentation")
    print(f"      - Result: Model generalizes better across domains")
    
    # 3. Architecture improvements
    print(f"\n   3. ARCHITECTURE IMPROVEMENTS:")
    print(f"      - Add residual connections")
    print(f"      - Use deeper but thinner network")
    print(f"      - Implement attention mechanisms")
    print(f"      - Add feature normalization layers")
    print(f"      - Result: Better feature extraction")
    
    # 4. Training strategy
    print(f"\n   4. IMPROVED TRAINING STRATEGY:")
    print(f"      - Increase epochs from 3 to 50+")
    print(f"      - Use learning rate scheduling")
    print(f"      - Implement early stopping")
    print(f"      - Use curriculum learning")
    print(f"      - Result: Better convergence and generalization")
    
    return "training_improvements"

def demonstrate_vectorization(features):
    """Demonstrate how to vectorize and re-rasterize training data"""
    
    print(f"\n🔬 VECTORIZATION DEMONSTRATION")
    print("=" * 35)
    
    if features is None:
        print("❌ No training data available for demonstration")
        return
    
    try:
        from skimage.morphology import skeletonize
        from skimage.measure import label, regionprops
        from scipy.ndimage import gaussian_filter
        
        # Take a sample
        sample = features[0].reshape(28, 28)
        
        # Step 1: Threshold to binary
        binary = sample > 0.5
        
        # Step 2: Skeletonize to get stroke centerlines
        skeleton = skeletonize(binary)
        
        # Step 3: Extract coordinate paths (simplified)
        # In practice, you'd need more sophisticated path extraction
        y_coords, x_coords = np.where(skeleton)
        coordinates = list(zip(x_coords, y_coords))
        
        # Step 4: Re-render using PIL (like your inference)
        img = Image.new('L', (28, 28), 0)
        draw = ImageDraw.Draw(img)
        
        if len(coordinates) > 1:
            # Draw connected lines
            for i in range(len(coordinates) - 1):
                draw.line([coordinates[i], coordinates[i+1]], fill=255, width=1)
        
        re_rendered = np.array(img) / 255.0
        
        # Visualize comparison
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        
        axes[0].imshow(sample, cmap='gray')
        axes[0].set_title('Original Training')
        axes[0].axis('off')
        
        axes[1].imshow(binary, cmap='gray')
        axes[1].set_title('Thresholded')
        axes[1].axis('off')
        
        axes[2].imshow(skeleton, cmap='gray')
        axes[2].set_title('Skeletonized')
        axes[2].axis('off')
        
        axes[3].imshow(re_rendered, cmap='gray')
        axes[3].set_title('Re-rendered')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig('vectorization_demo.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Vectorization demo saved as 'vectorization_demo.png'")
        print(f"   📊 Original stroke pixels: {np.sum(sample > 0.5)}")
        print(f"   📊 Skeleton pixels: {np.sum(skeleton)}")
        print(f"   📊 Re-rendered pixels: {np.sum(re_rendered > 0.5)}")
        
    except ImportError:
        print("   ❌ Skimage not available for vectorization demo")
    except Exception as e:
        print(f"   ❌ Error in vectorization demo: {e}")

def recommend_best_strategy():
    """Recommend the best strategy based on effort vs impact"""
    
    print(f"\n⭐ STRATEGY RECOMMENDATIONS")
    print("=" * 35)
    
    strategies = [
        {
            'name': 'Training Improvements',
            'effort': 'LOW',
            'impact': 'HIGH',
            'time': '1-2 days',
            'description': 'Fix training parameters + data augmentation'
        },
        {
            'name': 'Inference Adaptation',
            'effort': 'LOW',
            'impact': 'MEDIUM',
            'time': '0.5 days',
            'description': 'Modify canvas preprocessing to match training'
        },
        {
            'name': 'Training Data Preprocessing',
            'effort': 'MEDIUM',
            'impact': 'HIGH',
            'time': '2-3 days',
            'description': 'Vectorize and re-render training data'
        },
        {
            'name': 'Hybrid Approach',
            'effort': 'HIGH',
            'impact': 'HIGHEST',
            'time': '3-5 days',
            'description': 'Combine multiple strategies'
        }
    ]
    
    print(f"📊 Strategy Comparison:")
    print(f"{'Strategy':<25} {'Effort':<8} {'Impact':<8} {'Time':<10} {'Description'}")
    print("-" * 80)
    
    for strategy in strategies:
        print(f"{strategy['name']:<25} {strategy['effort']:<8} {strategy['impact']:<8} {strategy['time']:<10} {strategy['description']}")
    
    print(f"\n🎯 RECOMMENDED APPROACH:")
    print(f"   1. START WITH: Training Improvements (Low effort, High impact)")
    print(f"      - Fix epochs (3→50), dropout (0.6→0.3), add augmentation")
    print(f"      - This alone might solve 70-80% of the problem")
    
    print(f"\n   2. IF NEEDED: Inference Adaptation (Low effort, Medium impact)")
    print(f"      - Adjust canvas stroke thickness/blur")
    print(f"      - Match preprocessing exactly")
    
    print(f"\n   3. ADVANCED: Training Data Preprocessing (Medium effort, High impact)")
    print(f"      - Only if above approaches insufficient")
    print(f"      - Vectorize and re-render training data")

def main():
    print("🔄 DOMAIN ADAPTATION WITHOUT MANUAL DATA COLLECTION")
    print("=" * 60)
    print("Exploring practical solutions to handle domain shift\n")
    
    # Load and analyze current data
    features = analyze_domain_adaptation_options()
    
    # Explore different strategies
    strategy_1_preprocess_training_data(features)
    strategy_2_adapt_inference_pipeline(features)
    strategy_3_hybrid_approach()
    strategy_4_training_improvements()
    
    # Provide recommendations
    recommend_best_strategy()
    
    print(f"\n💡 KEY INSIGHT:")
    print("The training parameter issues (3 epochs, 0.6 dropout) are likely")
    print("causing MORE problems than the domain shift itself!")
    print("Start with training improvements - they're the easiest wins.")

if __name__ == "__main__":
    main()