import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64

st.set_page_config(
    page_title="ASL Test Image Generator",
    page_icon="🖼️",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .test-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def create_test_hand_image(letter, size=(300, 300), bg_color=(240, 240, 240)):
    """Create a simple test image for ASL letter"""
    
    # Create base image
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 120)
    except:
        font = ImageFont.load_default()
    
    # Add some hand-like shapes based on letter characteristics
    center_x, center_y = size[0] // 2, size[1] // 2
    
    if letter in ['A', 'E', 'M', 'N', 'S', 'T']:  # Closed fist letters
        # Draw closed fist shape
        draw.ellipse([center_x-60, center_y-80, center_x+60, center_y+40], 
                    fill=(205, 164, 133), outline=(139, 105, 82), width=3)
        draw.ellipse([center_x-10, center_y-60, center_x+20, center_y-30], 
                    fill=(205, 164, 133), outline=(139, 105, 82), width=2)  # Thumb
        
    elif letter in ['B', 'D', 'H', 'K', 'P']:  # Extended fingers
        # Draw palm
        draw.ellipse([center_x-50, center_y-20, center_x+50, center_y+80], 
                    fill=(205, 164, 133), outline=(139, 105, 82), width=3)
        # Draw fingers
        for i in range(4):
            x_offset = -30 + i * 20
            draw.rectangle([center_x+x_offset-5, center_y-80, center_x+x_offset+5, center_y-20], 
                         fill=(205, 164, 133), outline=(139, 105, 82), width=2)
        
    elif letter in ['C', 'O']:  # Curved shapes
        # Draw C or O shape
        draw.arc([center_x-70, center_y-70, center_x+70, center_y+70], 
                start=45, end=315, fill=(205, 164, 133), width=15)
        
    else:  # Complex formations
        # Draw general hand shape
        draw.ellipse([center_x-45, center_y-30, center_x+45, center_y+70], 
                    fill=(205, 164, 133), outline=(139, 105, 82), width=3)
        # Add some finger-like protrusions
        for i in range(3):
            x_offset = -20 + i * 20
            draw.ellipse([center_x+x_offset-8, center_y-50-i*5, center_x+x_offset+8, center_y-30-i*5], 
                        fill=(205, 164, 133), outline=(139, 105, 82), width=2)
    
    # Add letter text for reference
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = center_x - text_width // 2
    text_y = center_y + 100
    
    # Add text background
    draw.rectangle([text_x-10, text_y-10, text_x+text_width+10, text_y+text_height+10], 
                  fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw.text((text_x, text_y), letter, fill=(0, 0, 0), font=font)
    
    return img

def get_download_link(img, filename):
    """Generate download link for image"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">Download {filename}</a>'
    return href

def main():
    st.markdown('<h1 class="main-header">🖼️ ASL Test Image Generator</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="test-card">
        <h3>🎯 Generate Test Images for ASL Detection</h3>
        <p>Create simple hand-like test images for each ASL letter to test your detection apps</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ASL Classes
    ASL_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    # Sidebar controls
    st.sidebar.markdown("## 🎨 Image Settings")
    
    # Letter selection
    selected_letters = st.sidebar.multiselect(
        "Select letters to generate:",
        ASL_CLASSES,
        default=['A', 'B', 'C', 'D', 'E']
    )
    
    # Image settings
    image_size = st.sidebar.slider("Image Size", 200, 500, 300, 50)
    
    background_colors = {
        "Light Gray": (240, 240, 240),
        "White": (255, 255, 255),
        "Light Blue": (230, 240, 255),
        "Light Green": (240, 255, 240),
        "Beige": (245, 245, 220)
    }
    
    bg_choice = st.sidebar.selectbox("Background Color", list(background_colors.keys()))
    bg_color = background_colors[bg_choice]
    
    # Generation options
    generate_all = st.sidebar.button("🎨 Generate All 24 Letters", type="primary")
    generate_selected = st.sidebar.button("🖼️ Generate Selected Letters")
    
    # Quick test buttons
    st.sidebar.markdown("### 🚀 Quick Test")
    if st.sidebar.button("Generate A, B, C"):
        selected_letters = ['A', 'B', 'C']
        generate_selected = True
    
    if st.sidebar.button("Generate Fist Letters (A,E,M,N,S,T)"):
        selected_letters = ['A', 'E', 'M', 'N', 'S', 'T']
        generate_selected = True
    
    # Main content
    if generate_all:
        selected_letters = ASL_CLASSES
        
    if generate_all or generate_selected:
        if selected_letters:
            st.markdown(f"## 🖼️ Generated Test Images ({len(selected_letters)} letters)")
            
            # Generate images
            cols_per_row = 4
            for i in range(0, len(selected_letters), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j, letter in enumerate(selected_letters[i:i+cols_per_row]):
                    with cols[j]:
                        # Generate image
                        test_img = create_test_hand_image(
                            letter, 
                            size=(image_size, image_size),
                            bg_color=bg_color
                        )
                        
                        # Display image
                        st.image(test_img, caption=f"ASL Letter {letter}", use_column_width=True)
                        
                        # Download link
                        download_link = get_download_link(test_img, f"asl_test_{letter}.png")
                        st.markdown(download_link, unsafe_allow_html=True)
            
            # Batch download info
            st.markdown("---")
            st.info("💡 **How to use these images:**\n1. Download the test images\n2. Go to your ASL detection apps\n3. Upload these images to test character detection\n4. Compare results across different processing methods")
            
            # Links to detection apps
            st.markdown("""
            ### 🔗 Test Your Images Here:
            - **[Advanced Detection App](http://localhost:8504)** - Full processing pipeline
            - **[Comparison Lab](http://localhost:8505)** - Compare processing methods  
            - **[Demo App](http://localhost:8503)** - Simple testing
            """)
        else:
            st.warning("Please select at least one letter to generate.")
    
    else:
        # Instructions
        st.markdown("""
        ## 📋 How to Generate Test Images
        
        1. **Select Letters** from the sidebar (or use quick test buttons)
        2. **Adjust Settings** like image size and background color
        3. **Generate Images** using the buttons
        4. **Download** individual images or all at once
        5. **Test** in your ASL detection apps
        
        ### 🎯 Purpose of Test Images
        
        These simple test images help you:
        - **Test Detection Logic** - See if apps detect different characters
        - **Compare Processing Methods** - Test which image processing works best
        - **Validate Features** - Check if feature extraction works correctly
        - **Debug Issues** - Identify problems with specific letter types
        
        ### 🖼️ Image Types Generated
        
        - **Closed Fist Letters**: A, E, M, N, S, T (simple oval shapes)
        - **Extended Fingers**: B, D, H, K, P (rectangles for fingers)
        - **Curved Shapes**: C, O (arc shapes)
        - **Complex Letters**: All others (mixed shapes)
        
        Each image includes:
        - Hand-like shape based on letter characteristics
        - Letter label for reference
        - Skin-like color for realistic testing
        - Clean background for easy processing
        """)
        
        # Sample preview
        st.markdown("### 👀 Sample Preview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sample_a = create_test_hand_image('A', size=(200, 200))
            st.image(sample_a, caption="Sample: Letter A (Closed Fist)")
        
        with col2:
            sample_b = create_test_hand_image('B', size=(200, 200))
            st.image(sample_b, caption="Sample: Letter B (Extended Fingers)")
        
        with col3:
            sample_c = create_test_hand_image('C', size=(200, 200))
            st.image(sample_c, caption="Sample: Letter C (Curved Shape)")

if __name__ == "__main__":
    main()