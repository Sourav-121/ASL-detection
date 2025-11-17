"""
ASL Model Training Script
Based on the successful 94.57% accuracy model from the notebook
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
DATASET_PATH = "dataset"

# ASL Classes
ASL_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
NUM_CLASSES = len(ASL_CLASSES)

def load_dataset():
    """Load and preprocess dataset"""
    print("📁 Loading dataset...")
    
    # Get class folders
    class_folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
    class_folders.sort()
    
    all_images = []
    all_labels = []
    
    for i, folder in enumerate(class_folders):
        folder_path = os.path.join(DATASET_PATH, folder)
        print(f"Loading class {i+1}/{len(class_folders)}: {folder}")
        
        # Get all image files
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            
            try:
                # Load and preprocess image
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                image = image.astype(np.float32) / 255.0
                
                all_images.append(image)
                all_labels.append(i)
                
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
    
    print(f"✅ Loaded {len(all_images)} images from {len(class_folders)} classes")
    
    return np.array(all_images), np.array(all_labels)

def create_mobilenet_model():
    """Create MobileNetV2 transfer learning model"""
    print("🏗️ Creating MobileNetV2 model...")
    
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add classification head
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def create_advanced_cnn():
    """Create advanced CNN model"""
    print("🏗️ Creating Advanced CNN model...")
    
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # First block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)
    
    # Second block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Third block
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Fourth block
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classification head
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='advanced_cnn')
    return model

def create_data_generators(X_train, X_val, X_test, y_train, y_val, y_test):
    """Create data generators with augmentation"""
    print("🔄 Creating data generators...")
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation and test generators (no augmentation)
    val_test_datagen = ImageDataGenerator()
    
    # Create generators
    train_gen = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_gen = val_test_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    test_gen = val_test_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_gen, val_gen, test_gen

def train_model(model, train_gen, val_gen, model_name):
    """Train a model"""
    print(f"🚀 Training {model_name}...")
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            f'{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, test_gen, model_name):
    """Evaluate model performance"""
    print(f"📊 Evaluating {model_name}...")
    
    # Evaluate
    test_loss, test_accuracy, test_top3 = model.evaluate(test_gen, verbose=0)
    
    print(f"Test Results for {model_name}:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"  Top-3 Accuracy: {test_top3:.4f} ({test_top3*100:.2f}%)")
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_top3_accuracy': test_top3,
        'total_params': model.count_params()
    }

def plot_training_history(histories, model_names):
    """Plot training histories"""
    print("📈 Creating training plots...")
    
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy
    plt.subplot(2, 2, 1)
    for history, name in zip(histories, model_names):
        plt.plot(history.history['accuracy'], label=f'{name} - Train')
        plt.plot(history.history['val_accuracy'], label=f'{name} - Val')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(2, 2, 2)
    for history, name in zip(histories, model_names):
        plt.plot(history.history['loss'], label=f'{name} - Train')
        plt.plot(history.history['val_loss'], label=f'{name} - Val')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot top-3 accuracy
    plt.subplot(2, 2, 3)
    for history, name in zip(histories, model_names):
        if 'top_3_accuracy' in history.history:
            plt.plot(history.history['top_3_accuracy'], label=f'{name} - Train')
            plt.plot(history.history['val_top_3_accuracy'], label=f'{name} - Val')
    plt.title('Top-3 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Top-3 Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training pipeline"""
    print("🤟 ASL Alphabet Classification Training Pipeline")
    print("=" * 60)
    
    # Check if GPU is available
    if tf.config.experimental.list_physical_devices('GPU'):
        print("🚀 GPU detected! Training will be accelerated")
    else:
        print("💻 Using CPU for training")
    
    # Load dataset
    X, y = load_dataset()
    
    # Split dataset
    print("📊 Splitting dataset...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42  # 0.176 ≈ 0.15/0.85
    )
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)
    
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators(
        X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat
    )
    
    # Train models
    models_to_train = [
        (create_mobilenet_model(), "MobileNetV2"),
        (create_advanced_cnn(), "Advanced_CNN")
    ]
    
    histories = []
    results = {}
    
    for model, model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        # Show model summary
        print(f"📋 {model_name} Architecture:")
        model.summary()
        
        # Train model
        history = train_model(model, train_gen, val_gen, model_name)
        histories.append(history)
        
        # Evaluate model
        results[model_name] = evaluate_model(model, test_gen, model_name)
        
        # Save model in Keras format
        model.save(f'{model_name}_final.keras')
        print(f"💾 Model saved as {model_name}_final.keras")
    
    # Plot training histories
    plot_training_history(histories, [name for _, name in models_to_train])
    
    # Final comparison
    print("\n" + "="*60)
    print("🏆 FINAL MODEL COMPARISON")
    print("="*60)
    
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('test_accuracy', ascending=False)
    
    print("Model Performance Ranking:")
    for i, (model_name, row) in enumerate(comparison_df.iterrows()):
        rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else "🏅"
        print(f"{rank_emoji} {model_name}:")
        print(f"   Accuracy: {row['test_accuracy']:.4f} ({row['test_accuracy']*100:.2f}%)")
        print(f"   Top-3 Accuracy: {row['test_top3_accuracy']:.4f}")
        print(f"   Parameters: {row['total_params']:,}")
        print()
    
    # Identify best model
    best_model_name = comparison_df.index[0]
    best_accuracy = comparison_df.iloc[0]['test_accuracy']
    
    print(f"🎯 Best Model: {best_model_name}")
    print(f"🎯 Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    # Save comparison results
    comparison_df.to_csv('model_comparison_results.csv')
    print("💾 Results saved to model_comparison_results.csv")
    
    print("\n✅ Training pipeline completed successfully!")
    print("🤟 Ready for real-time ASL detection!")

if __name__ == "__main__":
    main()