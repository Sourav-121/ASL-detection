"""
Model Extraction and Conversion Script
Extract the best performing model from your notebook for use in the Streamlit app
"""

import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# ASL Classes
ASL_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

def recreate_best_model():
    """
    Recreate the best performing model architecture from your notebook
    Based on the 94.57% accuracy MobileNetV2 model
    """
    print("🏗️ Recreating best model architecture...")
    
    # MobileNetV2 base model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(128, 128, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Create the complete model with the same architecture as your notebook
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(24, activation='softmax')  # 24 ASL classes
    ])
    
    # Compile with same settings as notebook
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    print("✅ Model architecture recreated successfully!")
    return model

def convert_h5_to_keras(h5_path, keras_path):
    """Convert .h5 model to .keras format"""
    print(f"🔄 Converting {h5_path} to {keras_path}...")
    
    try:
        # Load the .h5 model
        model = load_model(h5_path)
        
        # Save in .keras format
        model.save(keras_path)
        print(f"✅ Model converted and saved to {keras_path}")
        return True
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        return False

def create_demo_model():
    """
    Create a demo model with random weights for testing
    Use this if you don't have a trained model yet
    """
    print("🎭 Creating demo model for testing...")
    
    model = recreate_best_model()
    
    # Save the demo model
    model.save('demo_asl_model.keras')
    print("✅ Demo model saved as 'demo_asl_model.keras'")
    print("⚠️ Note: This is a demo model with random weights!")
    print("   For actual use, please train the model or load your trained weights.")
    
    return model

def validate_model(model_path):
    """Validate that the model works correctly"""
    print(f"🔍 Validating model: {model_path}")
    
    try:
        # Load model
        model = load_model(model_path)
        
        # Check input shape
        expected_shape = (None, 128, 128, 3)
        actual_shape = model.input_shape
        
        if actual_shape == expected_shape:
            print("✅ Input shape is correct: (128, 128, 3)")
        else:
            print(f"⚠️ Input shape mismatch. Expected: {expected_shape}, Got: {actual_shape}")
        
        # Check output shape
        expected_classes = 24
        actual_classes = model.output_shape[-1]
        
        if actual_classes == expected_classes:
            print(f"✅ Output classes correct: {expected_classes}")
        else:
            print(f"⚠️ Output classes mismatch. Expected: {expected_classes}, Got: {actual_classes}")
        
        # Test prediction with dummy data
        dummy_input = np.random.random((1, 128, 128, 3)).astype(np.float32)
        prediction = model.predict(dummy_input, verbose=0)
        
        if prediction.shape == (1, 24):
            print("✅ Model prediction test passed")
        else:
            print(f"❌ Model prediction test failed. Output shape: {prediction.shape}")
        
        # Check if predictions sum to 1 (softmax output)
        pred_sum = np.sum(prediction[0])
        if 0.99 <= pred_sum <= 1.01:  # Allow small floating point errors
            print("✅ Softmax output validation passed")
        else:
            print(f"⚠️ Softmax output sum: {pred_sum} (should be close to 1.0)")
        
        print(f"✅ Model validation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False

def create_model_info(model_path):
    """Create a model info file with metadata"""
    try:
        model = load_model(model_path)
        
        model_info = {
            'classes': ASL_CLASSES,
            'num_classes': len(ASL_CLASSES),
            'input_shape': (128, 128, 3),
            'model_type': 'ASL Alphabet Classification',
            'preprocessing': {
                'resize': (128, 128),
                'normalize': 'divide_by_255',
                'color_format': 'RGB'
            },
            'total_parameters': model.count_params(),
            'architecture': 'MobileNetV2 + Custom Head'
        }
        
        # Save model info
        info_path = model_path.replace('.keras', '_info.pkl').replace('.h5', '_info.pkl')
        with open(info_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"📋 Model info saved to {info_path}")
        return model_info
        
    except Exception as e:
        print(f"❌ Error creating model info: {e}")
        return None

def main():
    """Main model extraction and conversion function"""
    print("🤖 ASL Model Extraction and Conversion Tool")
    print("="*50)
    
    # Check for existing model files
    model_files = []
    for file in os.listdir('.'):
        if file.endswith('.h5') or file.endswith('.keras'):
            model_files.append(file)
    
    if model_files:
        print(f"📁 Found {len(model_files)} model file(s):")
        for i, model in enumerate(model_files, 1):
            print(f"   {i}. {model}")
        
        print("\nChoose an option:")
        print("1. Validate existing model")
        print("2. Convert .h5 to .keras format")
        print("3. Create demo model for testing")
        print("Enter choice (1/2/3): ", end="")
        
        choice = input().strip()
        
        if choice == "1":
            print("\nSelect model to validate:")
            for i, model in enumerate(model_files, 1):
                print(f"{i}. {model}")
            print("Enter number: ", end="")
            
            try:
                idx = int(input().strip()) - 1
                if 0 <= idx < len(model_files):
                    selected_model = model_files[idx]
                    if validate_model(selected_model):
                        create_model_info(selected_model)
                else:
                    print("❌ Invalid selection")
            except ValueError:
                print("❌ Please enter a valid number")
        
        elif choice == "2":
            h5_files = [f for f in model_files if f.endswith('.h5')]
            if h5_files:
                print("\nSelect .h5 file to convert:")
                for i, model in enumerate(h5_files, 1):
                    print(f"{i}. {model}")
                print("Enter number: ", end="")
                
                try:
                    idx = int(input().strip()) - 1
                    if 0 <= idx < len(h5_files):
                        h5_file = h5_files[idx]
                        keras_file = h5_file.replace('.h5', '.keras')
                        if convert_h5_to_keras(h5_file, keras_file):
                            validate_model(keras_file)
                            create_model_info(keras_file)
                    else:
                        print("❌ Invalid selection")
                except ValueError:
                    print("❌ Please enter a valid number")
            else:
                print("❌ No .h5 files found")
        
        elif choice == "3":
            create_demo_model()
    
    else:
        print("📁 No model files found in current directory")
        print("\nChoose an option:")
        print("1. Create demo model for testing")
        print("2. Recreate model architecture (you'll need to load weights separately)")
        print("Enter choice (1/2): ", end="")
        
        choice = input().strip()
        
        if choice == "1":
            create_demo_model()
        elif choice == "2":
            model = recreate_best_model()
            model.save('asl_model_architecture.keras')
            print("✅ Model architecture saved as 'asl_model_architecture.keras'")
            print("💡 You can load your trained weights into this model structure")
    
    print("\n🎯 Model preparation completed!")
    print("\n💡 Next steps:")
    print("   1. Run 'python quick_start.py' to launch the app")
    print("   2. Or run 'streamlit run streamlit_app.py' directly")
    print("   3. For real-time detection: 'streamlit run realtime_app.py'")

if __name__ == "__main__":
    main()