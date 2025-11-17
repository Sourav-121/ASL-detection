import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os
from PIL import Image
import time
import mediapipe as mp
import warnings
import threading
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import asyncio

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="ASL Real-time Detection",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .prediction-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin: 20px 0;
        color: white;
        font-size: 3rem;
        font-weight: bold;
        box-shadow: 0 8px 32px 0 rgba(31,38,135,.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255,255,255,.18);
    }
    .confidence-meter {
        background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        height: 30px;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        margin: 10px 0;
    }
    .stats-container {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .hand-status {
        display: inline-block;
        padding: 10px 20px;
        border-radius: 25px;
        color: white;
        font-weight: bold;
        margin: 5px;
    }
    .hand-detected {
        background: linear-gradient(45deg, #56ab2f, #a8e6cf);
    }
    .hand-not-detected {
        background: linear-gradient(45deg, #ff416c, #ff4b2b);
    }
</style>
""", unsafe_allow_html=True)

# ASL Classes
ASL_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

class ASLRealTimePredictor:
    def __init__(self):
        self.model = None
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Prediction smoothing
        self.prediction_history = []
        self.max_history = 5
        
        # Statistics
        self.total_frames = 0
        self.hands_detected = 0
        self.predictions_made = 0
        
    def load_model_file(self, model_path):
        """Load trained model"""
        try:
            self.model = load_model(model_path)
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_hand_region(self, hand_region, target_size=(128, 128)):
        """Preprocess hand region for prediction"""
        try:
            # Resize image
            resized = cv2.resize(hand_region, target_size)
            
            # Normalize pixel values
            normalized = resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            batch_input = np.expand_dims(normalized, axis=0)
            
            return batch_input
        except Exception as e:
            return None
    
    def detect_and_predict(self, frame):
        """Detect hand and predict ASL letter"""
        self.total_frames += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        prediction_result = {
            'hand_detected': False,
            'prediction': 'No Hand',
            'confidence': 0.0,
            'top_3': [],
            'bbox': None
        }
        
        if results.multi_hand_landmarks:
            self.hands_detected += 1
            prediction_result['hand_detected'] = True
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks with style
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get bounding box
                h, w, _ = frame.shape
                x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Add padding
                padding = 40
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                prediction_result['bbox'] = (x_min, y_min, x_max, y_max)
                
                # Extract hand region
                hand_region = frame[y_min:y_max, x_min:x_max]
                
                if hand_region.size > 0 and self.model is not None:
                    # Preprocess and predict
                    processed_input = self.preprocess_hand_region(hand_region)
                    
                    if processed_input is not None:
                        predictions = self.model.predict(processed_input, verbose=0)
                        predicted_class_idx = np.argmax(predictions[0])
                        confidence = float(predictions[0][predicted_class_idx])
                        predicted_class = ASL_CLASSES[predicted_class_idx]
                        
                        # Get top 3 predictions
                        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                        top_3_predictions = [
                            (ASL_CLASSES[idx], float(predictions[0][idx])) 
                            for idx in top_3_idx
                        ]
                        
                        prediction_result.update({
                            'prediction': predicted_class,
                            'confidence': confidence,
                            'top_3': top_3_predictions
                        })
                        
                        self.predictions_made += 1
                        
                        # Add to history for smoothing
                        self.prediction_history.append((predicted_class, confidence))
                        if len(self.prediction_history) > self.max_history:
                            self.prediction_history.pop(0)
                
                # Draw bounding box
                if prediction_result['bbox']:
                    x_min, y_min, x_max, y_max = prediction_result['bbox']
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                    
                    # Draw prediction text on frame
                    if prediction_result['confidence'] > 0:
                        text = f"{prediction_result['prediction']}: {prediction_result['confidence']:.2%}"
                        cv2.putText(frame, text, (x_min, y_min - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame, prediction_result
    
    def get_smoothed_prediction(self, threshold=0.6):
        """Get smoothed prediction based on history"""
        if not self.prediction_history:
            return None, 0.0
        
        # Filter high confidence predictions
        high_conf_predictions = [
            (pred, conf) for pred, conf in self.prediction_history 
            if conf >= threshold
        ]
        
        if not high_conf_predictions:
            return None, 0.0
        
        # Count occurrences
        from collections import Counter
        pred_counts = Counter([pred for pred, _ in high_conf_predictions])
        most_common_pred = pred_counts.most_common(1)[0][0]
        
        # Calculate average confidence for most common prediction
        avg_confidence = np.mean([
            conf for pred, conf in high_conf_predictions 
            if pred == most_common_pred
        ])
        
        return most_common_pred, avg_confidence
    
    def get_statistics(self):
        """Get prediction statistics"""
        return {
            'total_frames': self.total_frames,
            'hands_detected': self.hands_detected,
            'predictions_made': self.predictions_made,
            'detection_rate': (self.hands_detected / self.total_frames * 100) if self.total_frames > 0 else 0,
            'prediction_rate': (self.predictions_made / self.hands_detected * 100) if self.hands_detected > 0 else 0
        }

# Initialize predictor
if 'predictor' not in st.session_state:
    st.session_state.predictor = ASLRealTimePredictor()
    st.session_state.current_prediction = 'No Hand'
    st.session_state.current_confidence = 0.0
    st.session_state.hand_detected = False

def video_frame_callback(frame):
    """Process video frame"""
    img = frame.to_ndarray(format="bgr24")
    
    # Process frame
    processed_frame, prediction_result = st.session_state.predictor.detect_and_predict(img)
    
    # Update session state
    st.session_state.current_prediction = prediction_result['prediction']
    st.session_state.current_confidence = prediction_result['confidence']
    st.session_state.hand_detected = prediction_result['hand_detected']
    st.session_state.top_3_predictions = prediction_result.get('top_3', [])
    
    return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

def main():
    st.markdown('<h1 class="main-header">🤟 Real-time ASL Alphabet Detection</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🎮 Controls")
    
    # Model loading
    st.sidebar.markdown("### 🤖 Model Loading")
    
    # Look for existing models
    model_files = [f for f in os.listdir('.') if f.endswith(('.h5', '.keras'))]
    
    if model_files:
        selected_model = st.sidebar.selectbox("Select model:", ["None"] + model_files)
        if selected_model != "None" and st.sidebar.button("Load Model"):
            if st.session_state.predictor.load_model_file(selected_model):
                st.sidebar.success(f"✅ Model loaded: {selected_model}")
                st.session_state.model_loaded = True
            else:
                st.sidebar.error("❌ Failed to load model")
    
    # Upload model
    uploaded_model = st.sidebar.file_uploader("Upload model:", type=['h5', 'keras'])
    if uploaded_model:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            tmp_file.write(uploaded_model.read())
            if st.session_state.predictor.load_model_file(tmp_file.name):
                st.sidebar.success("✅ Model loaded!")
                st.session_state.model_loaded = True
            os.unlink(tmp_file.name)
    
    # Settings
    st.sidebar.markdown("### ⚙️ Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.7, 0.05)
    show_stats = st.sidebar.checkbox("Show Statistics", True)
    show_top3 = st.sidebar.checkbox("Show Top 3 Predictions", True)
    
    # Main content
    if not hasattr(st.session_state, 'model_loaded') or not st.session_state.model_loaded:
        st.warning("🤖 Please load a trained ASL model to start detection")
        st.info("Upload your model file using the sidebar, then start the webcam stream")
        return
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Live Video Feed")
        
        # WebRTC configuration
        RTC_CONFIGURATION = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        # Start webcam stream
        webrtc_ctx = webrtc_streamer(
            key="asl-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    with col2:
        # Prediction Display
        st.markdown("### 🎯 Current Prediction")
        
        # Hand detection status
        if hasattr(st.session_state, 'hand_detected'):
            status_class = "hand-detected" if st.session_state.hand_detected else "hand-not-detected"
            status_text = "✋ Hand Detected" if st.session_state.hand_detected else "❌ No Hand"
            st.markdown(f'<div class="hand-status {status_class}">{status_text}</div>', 
                       unsafe_allow_html=True)
        
        # Current prediction display
        if hasattr(st.session_state, 'current_prediction'):
            prediction = st.session_state.current_prediction
            confidence = st.session_state.current_confidence
            
            if confidence >= confidence_threshold:
                st.markdown(f"""
                <div class="prediction-display">
                    {prediction}
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence meter
                confidence_percentage = confidence * 100
                st.markdown(f"""
                <div class="confidence-meter" style="background: linear-gradient(90deg, 
                    #ff9a9e 0%, #fecfef {confidence_percentage}%, #f0f0f0 {confidence_percentage}%);">
                    Confidence: {confidence:.1%}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; color: #666;">
                    Low Confidence<br>
                    <small>{prediction} ({confidence:.1%})</small>
                </div>
                """)
        
        # Top 3 predictions
        if show_top3 and hasattr(st.session_state, 'top_3_predictions'):
            st.markdown("### 🏆 Top 3 Predictions")
            for i, (letter, conf) in enumerate(st.session_state.top_3_predictions[:3]):
                medal = ["🥇", "🥈", "🥉"][i]
                st.write(f"{medal} **{letter}**: {conf:.1%}")
        
        # Statistics
        if show_stats:
            stats = st.session_state.predictor.get_statistics()
            st.markdown(f"""
            <div class="stats-container">
                <h4>📊 Statistics</h4>
                <p><strong>Total Frames:</strong> {stats['total_frames']}</p>
                <p><strong>Hands Detected:</strong> {stats['hands_detected']}</p>
                <p><strong>Detection Rate:</strong> {stats['detection_rate']:.1f}%</p>
                <p><strong>Predictions Made:</strong> {stats['predictions_made']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Smoothed prediction
        smoothed_pred, smoothed_conf = st.session_state.predictor.get_smoothed_prediction(confidence_threshold)
        if smoothed_pred:
            st.markdown(f"""
            ### 🎯 Smoothed Prediction
            **{smoothed_pred}** ({smoothed_conf:.1%})
            """)
    
    # ASL Reference
    with st.expander("🔤 ASL Alphabet Reference"):
        st.markdown("### Available Letters")
        # Create grid of letters
        cols = st.columns(8)
        for i, letter in enumerate(ASL_CLASSES):
            with cols[i % 8]:
                st.markdown(f"**{letter}**")
        
        st.markdown("**Note:** J and Z require motion and are not included in this static classifier.")
    
    # Tips and Instructions
    with st.expander("💡 Usage Tips"):
        st.markdown("""
        ### For Best Results:
        
        1. **Lighting**: Ensure good, even lighting
        2. **Background**: Use a plain, contrasting background
        3. **Hand Position**: Keep your hand clearly visible and centered
        4. **Stability**: Hold the sign steady for a few seconds
        5. **Distance**: Position your hand at arm's length from camera
        6. **Clear Signs**: Make clear, distinct letter formations
        
        ### Troubleshooting:
        - If no hand is detected, try adjusting lighting or hand position
        - Low confidence may indicate unclear hand positioning
        - The model works best with signs similar to the training data
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🤟 ASL Real-time Detection App | Built with Streamlit, MediaPipe & TensorFlow</p>
        <p>Model Performance: ~94.57% accuracy on test data</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()