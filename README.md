# 🤟 ASL Alphabet Real-time Detection App

একটি real-time ASL (American Sign Language) alphabet detection application যা Streamlit, TensorFlow, এবং MediaPipe ব্যবহার করে তৈরি করা হয়েছে। এই app আপনার trained model ব্যবহার করে webcam এর মাধ্যমে ASL alphabet signs detect করতে পারে।

## ✨ Features

### 🎯 Core Features
- **Real-time Detection**: Live webcam feed থেকে ASL alphabet detection
- **High Accuracy**: ~94.57% accuracy (আপনার notebook এর মতো)
- **Hand Detection**: MediaPipe দিয়ে automatic hand region detection
- **Multiple Input Modes**: 
  - 📸 Image upload
  - 🎥 Live camera feed
  - 📁 Batch processing
- **Smart Prediction Smoothing**: Multiple frames এর prediction smooth করে
- **Confidence Scoring**: Prediction confidence display

### 🎨 UI Features
- **Modern Interface**: Gradient backgrounds এবং smooth animations
- **Real-time Statistics**: Detection rate, prediction rate tracking
- **Top-3 Predictions**: Most likely predictions display
- **ASL Reference**: Built-in alphabet reference
- **Responsive Design**: Mobile-friendly layout

### 🔧 Technical Features
- **24 ASL Classes**: A-Y (excluding J and Z যেগুলো motion-based)
- **Advanced Preprocessing**: Auto hand region extraction
- **Model Compatibility**: .h5 এবং .keras format support
- **Error Handling**: Robust error handling এবং fallbacks

## 🚀 Quick Start

### 1. Installation

```bash
# Clone বা download করুন project
cd your-project-folder

# Dependencies install করুন
pip install -r requirements.txt

# অথবা manually install করুন:
pip install streamlit tensorflow opencv-python mediapipe streamlit-webrtc
```

### 2. Model Preparation

আপনার notebook থেকে trained model use করতে পারেন:

```python
# আপনার notebook থেকে model save করুন
model.save('asl_model.h5')  # অথবা 'asl_model.keras'
```

অথবা নতুন model train করুন:

```bash
python train_model.py
```

### 3. Run the App

#### Basic Streamlit App (Image Upload + Simulated Camera)
```bash
streamlit run streamlit_app.py
```

#### Advanced Real-time App (WebRTC Camera)
```bash
streamlit run realtime_app.py
```

## 📁 Project Structure

```
asl-detection-app/
├── streamlit_app.py          # Main Streamlit app (basic version)
├── realtime_app.py          # Advanced real-time app with WebRTC
├── train_model.py           # Model training script
├── requirements.txt         # Dependencies
├── README.md               # This file
├── dataset/                # Your ASL dataset
│   ├── A-samples/
│   ├── B-samples/
│   └── ...
└── models/                 # Trained models (created after training)
    ├── asl_model.h5
    └── asl_model.keras
```

## 🎮 How to Use

### 1. Load Model
- Sidebar থেকে existing model select করুন
- অথবা নতুন model file upload করুন
- Model load হলে green checkmark দেখাবে

### 2. Choose Detection Mode

#### 📸 Image Upload Mode
- "Choose an image" button click করুন
- ASL sign এর image upload করুন
- Instant prediction পাবেন confidence score সহ

#### 🎥 Live Camera Mode
- "Start Camera" button click করুন
- আপনার hand camera এর সামনে রাখুন
- Real-time predictions দেখুন

#### 📁 Batch Processing Mode
- Multiple images upload করুন
- Batch results table এ দেখুন
- Summary statistics পাবেন

### 3. Adjust Settings
- **Confidence Threshold**: Prediction confidence minimum level
- **Show Top 3**: Top 3 predictions display করা
- **Show Hand Region**: Detected hand region highlight করা

## 🔧 Model Training

নতুন model train করতে চাইলে:

```bash
python train_model.py
```

এই script:
- আপনার dataset load করবে
- MobileNetV2 এবং Advanced CNN model train করবে
- Best model automatically select করবে
- Training history plots create করবে

## 🎯 Performance Optimization

### For Better Detection:
1. **Good Lighting**: ভালো lighting নিশ্চিত করুন
2. **Plain Background**: Simple background ব্যবহার করুন
3. **Clear Signs**: Clear এবং distinct hand formations করুন
4. **Steady Hands**: Hand position stable রাখুন
5. **Optimal Distance**: Camera থেকে arms length distance রাখুন

### For Better Performance:
1. **GPU Support**: CUDA enable করুন fast inference এর জন্য
2. **Model Optimization**: TensorFlow Lite conversion consider করুন
3. **Batch Processing**: Multiple images একসাথে process করুন

## 📊 Technical Details

### Model Architecture
- **Base Models**: MobileNetV2, Advanced CNN
- **Input Size**: 128×128×3
- **Output**: 24 classes (A-Y excluding J, Z)
- **Preprocessing**: Normalization (0-1), resizing, hand region extraction

### Hand Detection
- **Library**: MediaPipe Hands
- **Detection Confidence**: 0.7 minimum
- **Tracking Confidence**: 0.5 minimum
- **Max Hands**: 1 (single hand detection)

### Prediction Smoothing
- **History Buffer**: Last 5 predictions
- **Threshold Filtering**: Only high-confidence predictions
- **Majority Voting**: Most common prediction selection

## 🔍 Troubleshooting

### Common Issues:

1. **Model Loading Error**
   ```
   Solution: Check model file path এবং format (.h5 or .keras)
   ```

2. **Camera Not Working**
   ```
   Solution: Browser permissions check করুন, HTTPS connection ensure করুন
   ```

3. **Low Detection Rate**
   ```
   Solution: Lighting improve করুন, hand position adjust করুন
   ```

4. **Slow Performance**
   ```
   Solution: GPU enable করুন, smaller input size try করুন
   ```

## 🛠️ Customization

### Add New Models:
```python
# streamlit_app.py তে নতুন model architecture add করুন
def create_custom_model():
    # Your custom model here
    return model
```

### Modify UI:
```python
# CSS styling customize করুন
st.markdown("""
<style>
    .custom-style {
        /* Your custom CSS */
    }
</style>
""", unsafe_allow_html=True)
```

### Extend Functionality:
- Audio feedback add করুন
- Letter sequence tracking implement করুন
- Multi-hand detection enable করুন
- Additional sign languages support করুন

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Performance optimization
- UI/UX enhancements
- Additional features
- Bug fixes
- Documentation improvements

## 📄 License

This project is open source. Feel free to use and modify according to your needs.

## 🙏 Acknowledgments

- **MediaPipe**: Hand detection functionality
- **TensorFlow**: Deep learning framework
- **Streamlit**: Web app framework
- **OpenCV**: Image processing
- **ASL Dataset**: Training data source

## 📞 Support

যদি কোনো সমস্যা হয় বা help প্রয়োজন হয়:
1. Error messages carefully read করুন
2. Requirements properly install করা আছে কিনা check করুন
3. Model file correctly loaded আছে কিনা verify করুন
4. Camera permissions granted আছে কিনা ensure করুন

---

## 🚀 Ready to Start?

```bash
# Install dependencies
pip install -r requirements.txt

# Train model (optional)
python train_model.py

# Run the app
streamlit run streamlit_app.py
```

Happy ASL Detection! 🤟"# ASL" 
