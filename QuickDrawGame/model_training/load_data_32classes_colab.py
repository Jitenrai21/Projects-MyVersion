"""
QuickDraw Dataset Loader - 32 Classes Edition
Optimized for Google Colab with memory management and batch processing

This script loads 32 different QuickDraw classes from .npy files
and prepares them for confidence-calibrated model training.

Features:
- Handles 32 classes instead of 15
- Memory-efficient batch processing
- Google Colab compatibility
- Downloadable pickle files
- Progress tracking with visual indicators
- Error handling and validation

Usage in Google Colab:
1. Mount Google Drive or upload dataset
2. Update dataset_path to your data location
3. Run the script to generate features_32classes and labels_32classes files
"""

import numpy as np
import os
import pickle
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
from google.colab import files
import zipfile

# Configuration for 32 classes
DATASET_PATH = "/content/drive/MyDrive/QuickDrawDataset_32classes"  # Update this path
MAX_SAMPLES_PER_CLASS = 10000  # Samples per class
TARGET_CLASSES = 32
BATCH_SIZE = 5000  # Process in batches to manage memory

# Expected 32 QuickDraw classes (you can modify this list based on your specific classes)
QUICKDRAW_32_CLASSES = [
    'airplane', 'apple', ' banana', 'bicycle', 'bowtie', 'bus', 'candle', 
    'car', 'cat', 'computer', 'dog', 'door', 'elephant', 'envelope', 'fish', 'flower', 'guitar', 
    'horse', 'house', 'ice cream', 'lightning', 'moon', 'mountain', 'rabbit', 'smiley face',
    'star', 'sun', 'tent', 'toothbrush', 'tree', 'truck', 'wristwatch'
]

def setup_colab_environment():
    """
    Set up Google Colab environment with necessary imports and configurations
    """
    print("🚀 Setting up Google Colab environment...")
    
    # Check GPU availability
    import tensorflow as tf
    print(f"📱 TensorFlow version: {tf.__version__}")
    print(f"🔥 GPU Available: {tf.config.list_physical_devices('GPU')}")
    
    # Mount Google Drive (uncomment if needed)
    # from google.colab import drive
    # drive.mount('/content/drive')
    # print("📁 Google Drive mounted successfully!")
    
    return True

def validate_dataset_structure(dataset_path):
    """
    Validate that the dataset has the expected structure and files
    """
    print(f"🔍 Validating dataset structure at: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        print("💡 Please update DATASET_PATH in the script")
        return False
    
    files = os.listdir(dataset_path)
    npy_files = [f for f in files if f.endswith('.npy')]
    
    print(f"📂 Found {len(npy_files)} .npy files")
    
    if len(npy_files) < TARGET_CLASSES:
        print(f"⚠️  Warning: Found {len(npy_files)} files, expected {TARGET_CLASSES}")
        print("🔧 Adjusting TARGET_CLASSES to match available files")
        return len(npy_files)
    
    # Display first few filenames for verification
    print("📋 Sample files found:")
    for i, file in enumerate(npy_files[:5]):
        print(f"   {i+1}. {file}")
    if len(npy_files) > 5:
        print(f"   ... and {len(npy_files)-5} more files")
    
    return len(npy_files)

def load_class_batch(file_path, class_id, max_samples=MAX_SAMPLES_PER_CLASS):
    """
    Load and process a single class with memory optimization
    """
    try:
        # Load the .npy file
        data = np.load(file_path)
        
        # Limit samples and normalize
        if len(data) > max_samples:
            data = data[:max_samples]
        
        # Normalize to [0, 1] range
        x_data = data.astype('float32') / 255.0
        
        # Create labels for this class
        y_data = np.full((len(data), 1), class_id, dtype='float32')
        
        return x_data, y_data
        
    except Exception as e:
        print(f"❌ Error loading {file_path}: {str(e)}")
        return None, None

def load_data_32_classes(dataset_path=DATASET_PATH, target_classes=TARGET_CLASSES):
    """
    Load QuickDraw data for 32 classes with memory-efficient processing
    """
    print(f"📥 Loading QuickDraw data for {target_classes} classes...")
    print(f"🎯 Max samples per class: {MAX_SAMPLES_PER_CLASS:,}")
    print(f"🔧 Using batch processing for memory efficiency")
    
    # Get all .npy files
    files = [f for f in os.listdir(dataset_path) if f.endswith('.npy')]
    
    # Limit to target number of classes
    files = files[:target_classes]
    actual_classes = len(files)
    
    print(f"📊 Processing {actual_classes} classes:")
    
    # Initialize lists for batch processing
    all_features = []
    all_labels = []
    
    # Process each class
    for class_id, filename in enumerate(tqdm(files, desc="Loading classes")):
        file_path = os.path.join(dataset_path, filename)
        class_name = filename.replace('.npy', '').replace('_', ' ')
        
        print(f"   {class_id + 1:2d}. {class_name}")
        
        # Load class data
        x_class, y_class = load_class_batch(file_path, class_id)
        
        if x_class is not None:
            all_features.append(x_class)
            all_labels.append(y_class)
            
            # Display shape info
            print(f"       Shape: {x_class.shape}, Samples: {len(x_class):,}")
        
        # Memory cleanup every few classes
        if (class_id + 1) % 10 == 0:
            gc.collect()
    
    # Combine all data
    print(f"\n🔄 Combining data from {len(all_features)} classes...")
    
    # Concatenate features and labels
    features = np.vstack(all_features)
    labels = np.vstack(all_labels)
    
    # Memory cleanup
    del all_features, all_labels
    gc.collect()
    
    print(f"✅ Data loading complete!")
    print(f"   Features shape: {features.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Total samples: {len(features):,}")
    print(f"   Classes: {actual_classes}")
    print(f"   Memory usage: ~{features.nbytes / 1024**2:.1f} MB")
    
    return features, labels, actual_classes

def save_data_with_compression(features, labels, num_classes):
    """
    Save processed data with compression and create downloadable files
    """
    print(f"\n💾 Saving processed data...")
    
    # Create metadata
    metadata = {
        'num_classes': num_classes,
        'samples_per_class': MAX_SAMPLES_PER_CLASS,
        'total_samples': len(features),
        'feature_shape': features.shape,
        'label_shape': labels.shape,
        'data_type': str(features.dtype),
        'classes_list': QUICKDRAW_32_CLASSES[:num_classes]
    }
    
    # Save features
    features_file = f"features_{num_classes}classes"
    with open(features_file, "wb") as f:
        pickle.dump(features, f, protocol=4)
    print(f"✅ Features saved: {features_file}")
    
    # Save labels
    labels_file = f"labels_{num_classes}classes"
    with open(labels_file, "wb") as f:
        pickle.dump(labels, f, protocol=4)
    print(f"✅ Labels saved: {labels_file}")
    
    # Save metadata
    metadata_file = f"metadata_{num_classes}classes.pkl"
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f, protocol=4)
    print(f"✅ Metadata saved: {metadata_file}")
    
    # Get file sizes
    features_size = os.path.getsize(features_file) / 1024**2
    labels_size = os.path.getsize(labels_file) / 1024**2
    metadata_size = os.path.getsize(metadata_file) / 1024**2
    
    print(f"\n📊 File sizes:")
    print(f"   Features: {features_size:.1f} MB")
    print(f"   Labels: {labels_size:.1f} MB")
    print(f"   Metadata: {metadata_size:.3f} MB")
    print(f"   Total: {features_size + labels_size + metadata_size:.1f} MB")
    
    return features_file, labels_file, metadata_file

def create_download_package():
    """
    Create a downloadable zip package for local use
    """
    print(f"\n📦 Creating downloadable package...")
    
    # Create zip file
    zip_filename = f"quickdraw_{TARGET_CLASSES}classes_dataset.zip"
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add data files
        for file in [f"features_{TARGET_CLASSES}classes", 
                    f"labels_{TARGET_CLASSES}classes", 
                    f"metadata_{TARGET_CLASSES}classes.pkl"]:
            if os.path.exists(file):
                zipf.write(file)
                print(f"   ✅ Added {file}")
    
    # Get zip size
    zip_size = os.path.getsize(zip_filename) / 1024**2
    print(f"📦 Package created: {zip_filename} ({zip_size:.1f} MB)")
    
    # Download in Colab
    try:
        files.download(zip_filename)
        print(f"⬇️  Download started: {zip_filename}")
    except:
        print(f"💡 Manual download: Right-click {zip_filename} in file browser")
    
    return zip_filename

def visualize_data_distribution(labels, num_classes):
    """
    Create visualization of class distribution
    """
    print(f"\n📊 Creating data distribution visualization...")
    
    # Count samples per class
    unique, counts = np.unique(labels, return_counts=True)
    
    # Create bar plot
    plt.figure(figsize=(15, 6))
    plt.bar(range(num_classes), counts, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.title(f'QuickDraw Dataset - Class Distribution ({num_classes} Classes)', fontsize=14, fontweight='bold')
    plt.xlabel('Class ID', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, count in enumerate(counts):
        plt.text(i, count + 50, f'{count:,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'class_distribution_{num_classes}classes.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Distribution plot saved: class_distribution_{num_classes}classes.png")
    print(f"📈 Statistics:")
    print(f"   Mean samples per class: {np.mean(counts):.0f}")
    print(f"   Min samples: {np.min(counts):,}")
    print(f"   Max samples: {np.max(counts):,}")
    print(f"   Std deviation: {np.std(counts):.0f}")

def main():
    """
    Main execution function
    """
    print("🎨 QuickDraw Dataset Loader - 32 Classes Edition")
    print("=" * 55)
    print("🎯 Optimized for Google Colab with confidence calibration")
    
    # Setup environment
    setup_colab_environment()
    
    # Validate dataset
    actual_classes = validate_dataset_structure(DATASET_PATH)
    if not actual_classes:
        return False
    
    # Update target classes if needed
    target_classes = min(TARGET_CLASSES, actual_classes)
    
    # Load data
    features, labels, num_classes = load_data_32_classes(DATASET_PATH, target_classes)
    
    # Visualize distribution
    visualize_data_distribution(labels, num_classes)
    
    # Save processed data
    features_file, labels_file, metadata_file = save_data_with_compression(features, labels, num_classes)
    
    # Create download package
    zip_file = create_download_package()
    
    print(f"\n🎉 Data preparation completed successfully!")
    print("=" * 55)
    print(f"📋 Summary:")
    print(f"   ✅ Classes processed: {num_classes}")
    print(f"   ✅ Total samples: {len(features):,}")
    print(f"   ✅ Feature shape: {features.shape}")
    print(f"   ✅ Ready for confidence-calibrated training!")
    
    print(f"\n🔄 Next Steps:")
    print(f"   1. Use the generated files in training notebook")
    print(f"   2. Download {zip_file} for local use")
    print(f"   3. Update training script to use {num_classes} classes")
    print(f"   4. Run confidence-calibrated training!")
    
    return True

# Run the main function
if __name__ == "__main__":
    main()