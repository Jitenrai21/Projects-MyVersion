import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

def create_improved_model(image_x, image_y):
    """Create improved model with better architecture and regularization"""
    num_of_classes = 15
    model = Sequential()
    
    # First conv block with batch normalization
    model.add(Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    # Second conv block
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    # Dense layers with reduced dropout
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))  # Reduced from 0.6
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))  # Reduced from 0.6
    model.add(Dense(num_of_classes, activation='softmax'))

    # Better optimizer and compilation
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    # Improved callbacks
    filepath = "model_trad/QuickDraw_improved.keras"
    callbacks_list = [
        ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, 
                       save_best_only=True, mode='max'),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    ]

    return model, callbacks_list

def create_data_augmentation():
    """Create data augmentation generator for domain robustness"""
    datagen = ImageDataGenerator(
        rotation_range=15,      # Random rotations ±15°
        width_shift_range=0.1,  # Random horizontal shifts
        height_shift_range=0.1, # Random vertical shifts
        zoom_range=0.1,         # Random zoom ±10%
        shear_range=0.1,        # Random shear transformations
        fill_mode='constant',   # Fill with black (0)
        cval=0
    )
    return datagen

def load_and_preprocess_data():
    """Load and preprocess data with better practices"""
    # Load data
    with open("features_onTrad", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels_onTrad", "rb") as f:
        labels = np.array(pickle.load(f))
    
    print(f"Loaded data shapes: {features.shape}, {labels.shape}")
    
    # Shuffle data
    features, labels = shuffle(features, labels, random_state=42)
    
    # Convert labels to categorical
    labels = tf.keras.utils.to_categorical(labels)
    
    # Split data with proper validation set
    train_x, temp_x, train_y, temp_y = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    val_x, test_x, val_y, test_y = train_test_split(
        temp_x, temp_y, test_size=0.5, random_state=42, stratify=temp_y
    )
    
    # Reshape for CNN
    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
    val_x = val_x.reshape(val_x.shape[0], 28, 28, 1)
    test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
    
    print(f"Split sizes - Train: {len(train_x)}, Val: {len(val_x)}, Test: {len(test_x)}")
    
    return train_x, val_x, test_x, train_y, val_y, test_y

def train_improved_model():
    """Train the model with improved parameters"""
    
    print("🚀 TRAINING IMPROVED QUICKDRAW MODEL")
    print("=" * 40)
    
    # Load and preprocess data
    train_x, val_x, test_x, train_y, val_y, test_y = load_and_preprocess_data()
    
    # Create model and callbacks
    model, callbacks = create_improved_model(28, 28)
    
    print("\n📊 Model Architecture:")
    model.summary()
    
    # Create data augmentation
    datagen = create_data_augmentation()
    datagen.fit(train_x)
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Epochs: 50 (with early stopping)")
    print(f"   Batch size: 64")
    print(f"   Dropout: 0.3 (reduced from 0.6)")
    print(f"   Data augmentation: Enabled")
    print(f"   Learning rate: 0.001 with scheduling")
    print(f"   Validation split: Proper 70/10/20 split")
    
    # Train model with data augmentation
    print(f"\n🏃 Starting training...")
    history = model.fit(
        datagen.flow(train_x, train_y, batch_size=64),
        validation_data=(val_x, val_y),
        steps_per_epoch=len(train_x) // 64,
        epochs=50,  # Increased from 3
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print(f"\n📊 Final Evaluation:")
    test_loss, test_acc = model.evaluate(test_x, test_y, verbose=0)
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Save final model
    model.save('model_trad/QuickDraw_improved_final.keras')
    print(f"   ✅ Model saved as 'QuickDraw_improved_final.keras'")
    
    return model, history

if __name__ == "__main__":
    train_improved_model()