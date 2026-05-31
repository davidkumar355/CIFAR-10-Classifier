import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import inspect
from PIL import Image
import plotly.graph_objects as go
import os

# --- Page Configurations & Styling ---
st.set_page_config(
    page_title="CIFAR-10 EfficientNetV2-M Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Glassmorphic / Dark-Inspired Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Typography overrides */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Header Gradient */
    .title-gradient {
        background: linear-gradient(135deg, #FF6B6B 0%, #4D96FF 50%, #6BCB77 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #7E8B9B;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Card Container style */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(77, 150, 255, 0.4);
    }
    
    /* Sidebar Details Container */
    .sidebar-portfolio {
        background: rgba(77, 150, 255, 0.08);
        border: 1px solid rgba(77, 150, 255, 0.15);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 20px;
    }
    
    .sidebar-portfolio-title {
        color: #4D96FF;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .sidebar-portfolio-desc {
        font-size: 0.85rem;
        line-height: 1.4;
        color: #B5C0CD;
    }
    
    /* Stat Badge */
    .stat-badge {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(107, 203, 119, 0.15);
        border: 1px solid rgba(107, 203, 119, 0.3);
        color: #6BCB77;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    .stat-badge-blue {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(77, 150, 255, 0.15);
        border: 1px solid rgba(77, 150, 255, 0.3);
        color: #4D96FF;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 15px;
        margin-right: 8px;
    }
    
</style>
""", unsafe_allow_html=True)

# --- Model Loading & Caching ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_EfficientNetV2M_model.keras")

@st.cache_resource
def load_efficientnet_model():
    """Loads the pre-trained EfficientNetV2M model exactly once into memory."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found in the workspace directory.")
        st.stop()
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

with st.spinner("🚀 Initializing EfficientNetV2-M (CIFAR-10) Deep Neural Network..."):
    model = load_efficientnet_model()

# CIFAR-10 Class Names and Mapping Emojis
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
EMOJI_MAP = {
    'airplane': "✈️ Airplane",
    'automobile': "🚗 Automobile",
    'bird': "🐦 Bird",
    'cat': "🐱 Cat",
    'deer': "🦌 Deer",
    'dog': "🐶 Dog",
    'frog': "🐸 Frog",
    'horse': "🐴 Horse",
    'ship': "🚢 Ship",
    'truck': "🚚 Truck"
}

# --- Sidebar Component ---
with st.sidebar:
    st.markdown("<div class='sidebar-portfolio'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-portfolio-desc'>Welcome to the CIFAR-10 deep learning diagnostic portal. This web platform integrates a state-of-the-art <b>EfficientNetV2-M</b> convolutional neural network with <b>Grad-CAM (Gradient-weighted Class Activation Mapping)</b> interpretability to deliver real-time visual justifications of model predictions.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.subheader("🛠️ Settings & Diagnostics")
    
    # Pre-loaded sample images selector to improve UX immensely
    st.markdown("##### 📌 Test with Pre-loaded Images")
    test_dir = os.path.join(SCRIPT_DIR, "Test Images")
    sample_image = None
    if os.path.exists(test_dir):
        test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if test_files:
            selected_file = st.selectbox(
                "Select a sample image to classify:",
                ["None"] + sorted(test_files)
            )
            if selected_file != "None":
                sample_image = os.path.join(test_dir, selected_file)
                
    st.markdown("---")
    st.markdown("##### 📤 Upload Your Own Image")
    uploaded_file = st.file_uploader(
        "Supported formats: JPEG, JPG, PNG",
        type=['jpg', 'jpeg', 'png']
    )

# Determine the image source
img_to_process = None
if uploaded_file is not None:
    img_to_process = Image.open(uploaded_file)
elif sample_image is not None:
    img_to_process = Image.open(sample_image)

# --- Main Dashboard Header ---
st.markdown("<div class='title-gradient'>CIFAR-10 Interpretability Portal</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>EfficientNetV2-M Convolutional Neural Network with Diagnostic Grad-CAM Heatmap Overlays</div>", unsafe_allow_html=True)

# Display a beautiful empty state if no image is selected
if img_to_process is None:
    st.markdown("""
    <div class='glass-card' style='text-align: center; padding: 50px 20px;'>
        <h3 style='color: #4D96FF; margin-bottom: 10px;'>📊 Standby for Inference</h3>
        <p style='color: #8E9BAE; font-size: 1.05rem; max-width: 600px; margin: 0 auto 30px auto;'>
            To trigger predictions and visual explanations, please <b>upload an image</b> in the sidebar or <b>select one of the pre-loaded CIFAR-10 test samples</b>.
        </p>
        <div style='display: flex; justify-content: center; gap: 15px;'>
            <span class='stat-badge-blue'>🌐 Pre-trained on ImageNet</span>
            <span class='stat-badge'>🎯 Test Accuracy: 95.52%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # --- Inference and Diagnostics Execution ---
    
    # 1. PREPROCESSING CONSTRAINT (CRITICAL):
    # - Model contains internal Resizing(224, 224) and Rescaling layers.
    # - We must strictly resize to (32, 32) using PIL/OpenCV, convert to float32,
    #   and keep the pixel range strictly [0, 255] (no manual division by 255.0).
    # - Shape must be (1, 32, 32, 3).
    
    # Ensure RGB conversion (dropping alpha channel if present)
    img_rgb = img_to_process.convert("RGB")
    
    # Resize strictly to (32, 32) using PIL's bilinear interpolation
    resized_pil = img_rgb.resize((32, 32), Image.Resampling.BILINEAR)
    img_raw = np.array(resized_pil, dtype=np.float32) # Shape: (32, 32, 3), range [0, 255]
    
    # Final model input tensor scaled to [0, 1] to compensate for the model's internal Rescaling(scale=255.0) layer
    outer_input = np.expand_dims(img_raw, axis=0) / 255.0
    
    with st.spinner("⚡ Running EfficientNetV2-M Forward Pass..."):
        # Make predictions
        predictions = model.predict(outer_input, verbose=0)[0]
        
    # Extract prediction details
    top3_indices = np.argsort(predictions)[::-1][:3]
    top3_probs = predictions[top3_indices]
    top3_classes = [CLASS_NAMES[idx] for idx in top3_indices]
    
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    # --- Two Column Main Layout ---
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("### 📷 Input Specimen")
        # Display the uploaded image cleanly
        st.image(img_to_process, use_container_width=True, caption="Uploaded Original Image")
        st.markdown(f"""
        <div style='display: flex; gap: 8px; margin-top: 15px;'>
            <span class='stat-badge-blue'>Original Dimensions: {img_to_process.size[0]}x{img_to_process.size[1]}</span>
            <span class='stat-badge'>Diagnostic Resolution: 32x32</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Model Inference (Top 3)")
        
        # Highlighting the winner
        emoji_pred = EMOJI_MAP.get(predicted_class, predicted_class)
        st.success(f"**Predicted Category:** {emoji_pred} ({confidence:.2f}% Confidence)")
        
        # Plotly horizontal bar chart of probabilities
        fig = go.Figure()
        
        # Mapping nice labels
        probs_pct = top3_probs * 100
        labels_with_emojis = [EMOJI_MAP.get(cls, cls) for cls in top3_classes]
        
        fig.add_trace(go.Bar(
            x=probs_pct[::-1],
            y=labels_with_emojis[::-1],
            orientation='h',
            marker=dict(
                color=probs_pct[::-1],
                colorscale=[[0, '#4D96FF'], [0.5, '#6BCB77'], [1.0, '#FF6B6B']],
                line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
            ),
            text=[f"{val:.2f}%" for val in probs_pct[::-1]],
            textposition='auto',
            textfont=dict(family='Plus Jakarta Sans', size=12, color='white')
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            xaxis=dict(
                title="Probability (%)",
                range=[0, 100],
                showgrid=False,
                zeroline=False,
                title_font=dict(color='#8E9BAE', size=12),
                tickfont=dict(color='#8E9BAE')
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(color='white', size=13, family='Plus Jakarta Sans')
            ),
            margin=dict(l=20, r=20, t=10, b=10),
            height=280
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Grad-CAM Interpretability Section ---
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 4. PASTE THIS EXACT FUNCTION INTO THE SCRIPT
    def get_efficientnet_gradcam_heatmap(img_array, model, last_conv_layer_name='top_conv'):
        efficientnet_base = model.get_layer('efficientnetv2-m')
        
        efficientnet_grad_model = tf.keras.Model(
            inputs  = efficientnet_base.input,
            outputs = [
                efficientnet_base.get_layer(last_conv_layer_name).output,  
                efficientnet_base.output                                   
            ]
        )

        with tf.GradientTape() as tape:
            x = tf.cast(img_array, tf.float32)
            conv_outputs = None

            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                if layer.name == 'efficientnetv2-m':
                    conv_outputs, x = efficientnet_grad_model(x, training=False)
                    tape.watch(conv_outputs)
                else:
                    sig = inspect.signature(layer.call)
                    if 'training' in sig.parameters:
                        x = layer(x, training=False)
                    else:
                        x = layer(x)

            predictions = x  
            top_pred_index = tf.argmax(predictions[0])
            class_channel  = predictions[:, top_pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val == 0:
            max_val = tf.constant(1e-10)
        heatmap = heatmap / max_val
        return heatmap.numpy()

    with st.expander("🔍 Model Interpretability (Grad-CAM Diagnostic)", expanded=True):
        st.markdown("""
        <p style='color: #8E9BAE; font-size: 0.95rem; margin-bottom: 20px;'>
            <b>Grad-CAM (Gradient-weighted Class Activation Mapping)</b> utilizes gradients of the winning score flowing into the final convolutional layer (<code>top_conv</code> of the EfficientNetV2-M base model) to construct a spatial localization map. The resulting heatmap highlights the visual regions containing discriminative patterns most influential in determining the classification.
        </p>
        """, unsafe_allow_html=True)
        
        with st.spinner("🧠 Computation of Grad-CAM Activation Heatmaps..."):
            try:
                # Calculate raw heatmap
                heatmap = get_efficientnet_gradcam_heatmap(outer_input, model, last_conv_layer_name='top_conv')
                
                # Resize the 32x32 preprocessed image back to a premium display resolution (e.g., 224x224)
                display_size = (300, 300)
                
                # Base display image (original processed 32x32 scaled to display size)
                img_display = cv2.resize(img_raw, display_size, interpolation=cv2.INTER_LANCZOS4)
                img_display = np.clip(img_display, 0, 255).astype("uint8")
                
                # Resize heatmap from convolutional output spatial dimension to display size
                heatmap_resized = cv2.resize(heatmap, display_size, interpolation=cv2.INTER_CUBIC)
                heatmap_uint8 = np.uint8(255 * heatmap_resized)
                
                # Colorize the heatmap using the OpenCV Jet colormap
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                
                # Blending overlay (Superimposed image)
                alpha = 0.4
                superimposed_img = cv2.addWeighted(heatmap_color, alpha, img_display, 1.0 - alpha, 0.0)
                superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
                
                # Create visual columns
                c_cam1, c_cam2, c_cam3 = st.columns([1, 1, 1], gap="medium")
                
                with c_cam1:
                    st.image(img_display, use_container_width=True, caption="Model Perspective (Resized Input)")
                with c_cam2:
                    st.image(heatmap_color, use_container_width=True, caption="Gradient Class Activation Heatmap")
                with c_cam3:
                    st.image(superimposed_img, use_container_width=True, caption=f"Combined Diagnostic Overlay (Pred: {predicted_class})")
                
                # Interpretability Analysis card
                st.markdown(f"""
                <div class='glass-card' style='margin-top: 20px; background: rgba(77, 150, 255, 0.03); border: 1px dashed rgba(255,255,255,0.15);'>
                    <h4 style='color: #6BCB77;'>💡 Interpretability Findings</h4>
                    <p style='color: #B5C0CD; font-size: 0.92rem; margin: 0;'>
                        The warm (red/orange) zones on the diagnostic overlay mark the spatial coordinates inside the image that strongly triggered activations in the <code>top_conv</code> layer of <code>EfficientNetV2-M</code>. If the model predicted a <b>{predicted_class}</b>, you will observe the warm concentrations targeting its primary semantic components (e.g., the silhouette, eyes, wings, wheels, or hulls).
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as cam_err:
                st.error(f"Grad-CAM extraction failed: {str(cam_err)}")
