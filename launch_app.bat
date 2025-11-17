@echo off
echo ===============================================
echo 🤟 ASL Alphabet Detection App Launcher
echo ===============================================
echo.

echo 📦 Installing dependencies...
pip install streamlit tensorflow opencv-python mediapipe pandas numpy pillow matplotlib

echo.
echo ⚠️ Note: If you want real-time webcam features, also install:
echo pip install streamlit-webrtc aiortc av
echo.

echo 🤖 Checking for model files...
if exist *.h5 (
    echo ✅ Found .h5 model files
) else if exist *.keras (
    echo ✅ Found .keras model files  
) else (
    echo ⚠️ No model files found. You'll need to:
    echo    1. Upload a model in the app, OR
    echo    2. Train a new model using train_model.py
    echo.
)

echo 🚀 Choose which app to launch:
echo 1. Basic App (Image upload + Simulated camera)
echo 2. Advanced App (Real-time webcam - requires additional packages)
echo 3. Train new model first
echo.
set /p choice="Enter your choice (1/2/3): "

if "%choice%"=="1" (
    echo 🌟 Launching Basic Streamlit App...
    echo 📝 Your browser will open at http://localhost:8501
    streamlit run streamlit_app.py
) else if "%choice%"=="2" (
    echo 🌟 Launching Advanced Real-time App...
    echo 📝 Your browser will open at http://localhost:8501
    echo 🎥 Make sure to allow camera permissions!
    streamlit run realtime_app.py
) else if "%choice%"=="3" (
    echo 🏋️ Starting model training...
    python train_model.py
    pause
    echo 🚀 Now launching the app...
    streamlit run streamlit_app.py
) else (
    echo ❌ Invalid choice. Please run the script again.
)

pause