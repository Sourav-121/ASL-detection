"""
Quick Start Script for ASL Detection App
Run this to quickly set up and test the application
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print application banner"""
    print("\n" + "="*60)
    print("🤟 ASL Alphabet Detection App - Quick Start")
    print("="*60)
    print("Setting up your real-time ASL detection application...")
    print()

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", platform.python_version())
        return False
    print(f"✅ Python version: {platform.python_version()}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_dataset():
    """Check if dataset exists"""
    print("\n📁 Checking dataset...")
    if os.path.exists("dataset"):
        class_folders = [f for f in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", f))]
        if class_folders:
            print(f"✅ Dataset found with {len(class_folders)} classes")
            return True
        else:
            print("⚠️ Dataset folder exists but is empty")
    else:
        print("⚠️ Dataset folder not found")
    
    print("\n💡 To use the app, you need:")
    print("   1. Your ASL dataset in 'dataset' folder")
    print("   2. Or a pre-trained model file (.h5 or .keras)")
    return False

def check_models():
    """Check for existing model files"""
    print("\n🤖 Checking for model files...")
    model_files = []
    for file in os.listdir("."):
        if file.endswith((".h5", ".keras")) and "model" in file.lower():
            model_files.append(file)
    
    if model_files:
        print(f"✅ Found {len(model_files)} model file(s):")
        for model in model_files:
            print(f"   - {model}")
        return True
    else:
        print("⚠️ No pre-trained model files found")
        return False

def run_training():
    """Ask user if they want to train a new model"""
    if not check_dataset():
        return False
    
    print("\n🏋️ Do you want to train a new model? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice in ['y', 'yes']:
        print("\n🚀 Starting model training...")
        try:
            subprocess.run([sys.executable, "train_model.py"])
            print("✅ Model training completed!")
            return True
        except Exception as e:
            print(f"❌ Error during training: {e}")
            return False
    return False

def launch_app():
    """Launch the Streamlit app"""
    print("\n🚀 Choose app version to launch:")
    print("1. Basic App (Image upload + Simulated camera)")
    print("2. Advanced App (Real-time webcam with WebRTC)")
    print("3. Train new model first")
    print("Enter choice (1/2/3): ", end="")
    
    choice = input().strip()
    
    if choice == "1":
        print("\n🌟 Launching Basic Streamlit App...")
        print("📝 Open browser and go to: http://localhost:8501")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    
    elif choice == "2":
        print("\n🌟 Launching Advanced Real-time App...")
        print("📝 Open browser and go to: http://localhost:8501")
        print("🎥 Make sure to allow camera permissions!")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "realtime_app.py"])
    
    elif choice == "3":
        if run_training():
            launch_app()  # Recursively call after training
    
    else:
        print("❌ Invalid choice. Please run the script again.")

def main():
    """Main quick start function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install requirements
    if not install_requirements():
        print("\n💡 Try installing manually:")
        print("   pip install streamlit tensorflow opencv-python mediapipe")
        return
    
    # Check for dataset and models
    has_dataset = check_dataset()
    has_models = check_models()
    
    if not has_dataset and not has_models:
        print("\n❌ Neither dataset nor pre-trained models found!")
        print("\n💡 Please:")
        print("   1. Add your ASL dataset to 'dataset' folder, OR")
        print("   2. Place a pre-trained model file (.h5 or .keras) in current directory")
        return
    
    # Launch app
    launch_app()

if __name__ == "__main__":
    main()