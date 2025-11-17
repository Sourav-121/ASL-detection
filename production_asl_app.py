import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
import json
import os

# Load production model
@st.cache_resource
def load_production_model():
    try:
        model = joblib.load('asl_production_model.pkl')
        scaler = joblib.load('asl_production_scaler.pkl')
        
        with open('asl_model_info.json', 'r') as f:
            model_info = json.load(f)
        
        return model, scaler, model_info
    except Exception as e:
        st.error(f"Failed to load production model: {e}")
        return None, None, None

def extract_features(image):
    """Extract the same features used in training"""
    # Resize to standard size
    image = cv2.resize(image, (128, 128))
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    features = []
    
    # 1. Basic image statistics
    features.extend([
        np.mean(gray),
        np.std(gray),
        np.min(gray),
        np.max(gray),
        np.median(gray)
    ])
    
    # 2. Histogram features
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    hist_normalized = hist.flatten() / np.sum(hist)
    features.extend(hist_normalized)
    
    # 3. Edge features
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
    features.append(edge_density)
    
    # 4. Gradient features
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    features.extend([
        np.mean(magnitude),
        np.std(magnitude)
    ])
    
    # 5. Contour features
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        area = cv2.contourArea(largest_contour)
        features.append(area / (image.shape[0] * image.shape[1]))
        
        perimeter = cv2.arcLength(largest_contour, True)
        features.append(perimeter / (2 * (image.shape[0] + image.shape[1])))
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        features.extend([
            w / image.shape[1],
            h / image.shape[0], 
            float(w) / h if h != 0 else 0
        ])
    else:
        features.extend([0, 0, 0, 0, 0])
    
    # 6. Texture features
    diff_h = np.abs(gray[:, 1:] - gray[:, :-1])
    diff_v = np.abs(gray[1:, :] - gray[:-1, :])
    features.extend([
        np.mean(diff_h),
        np.std(diff_h),
        np.mean(diff_v),
        np.std(diff_v)
    ])
    
    # 7. Shape moments
    moments = cv2.moments(gray)
    if moments['m00'] != 0:
        cx = moments['m10'] / moments['m00'] / image.shape[1]
        cy = moments['m01'] / moments['m00'] / image.shape[0]
        features.extend([cx, cy])
        
        hu_moments = cv2.HuMoments(moments)
        features.extend([hu_moments[i][0] for i in range(3)])
    else:
        features.extend([0, 0, 0, 0, 0])
    
    # Ensure consistent feature count
    target_length = 50
    if len(features) < target_length:
        features.extend([0] * (target_length - len(features)))
    elif len(features) > target_length:
        features = features[:target_length]
    
    return np.array(features)

def main():
    st.set_page_config(page_title="Production ASL Detection", layout="wide")
    
    st.title("🏭 Production ASL Detection")
    st.markdown("**Trained on your actual dataset with advanced feature extraction**")
    
    # Load model
    model, scaler, model_info = load_production_model()
    
    if model is None:
        st.error("Production model not found! Please run the model extractor first.")
        return
    
    # Display model info
    with st.expander("🔍 Model Information"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Classes", len(model_info['classes']))
        with col2:
            st.metric("Features", model_info['n_features'])
        with col3:
            st.metric("Trees", model_info['n_estimators'])
    
    # Upload interface
    uploaded_file = st.file_uploader("Upload ASL Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image")
            
            # Convert to array for processing
            img_array = np.array(image)
            
        with col2:
            # Extract features and predict
            features = extract_features(img_array)
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Get predictions
            probabilities = model.predict_proba(features_scaled)[0]
            predicted_class = model.predict(features_scaled)[0]
            confidence = np.max(probabilities)
            
            # Display results
            st.markdown(f"""
            ### 🎯 Detection Result
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
                <h1 style="font-size: 4rem; margin: 0;">{predicted_class}</h1>
                <h3>{confidence*100:.1f}% Confident</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Top 3 predictions
            top_3_idx = np.argsort(probabilities)[-3:][::-1]
            st.markdown("### 🏆 Top 3 Predictions")
            
            for i, idx in enumerate(top_3_idx):
                char = model_info['classes'][idx]
                prob = probabilities[idx]
                medal = ["🥇", "🥈", "🥉"][i]
                
                st.markdown(f"{medal} **{char}** - {prob*100:.1f}%")
                st.progress(prob)

if __name__ == "__main__":
    main()