"""
Pest Sound Classification Dashboard
Run with: streamlit run dashboard.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64
from datetime import datetime
import time

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from config import Config
from predict import load_files, extract_features
import tensorflow as tf
import librosa
import librosa.display
import pickle

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Pest Sound Classifier",
    page_icon="🐞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-top: 0;
        padding-top: 0;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .confidence-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ED6A02;
        font-weight: bold;
    }
    .confidence-low {
        color: #D32F2F;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1B5E20;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'encoder' not in st.session_state:
    st.session_state.encoder = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'audio_files' not in st.session_state:
    st.session_state.audio_files = []

# ============================================
# HELPER FUNCTIONS
# ============================================
@st.cache_resource
def load_model_resources():
    """Load model and encoder with caching"""
    try:
        model_path = os.path.join(Config.MODEL_PATH, "final_model.keras")
        encoder_path = os.path.join(Config.MODEL_PATH, "label_encoder.pkl")
        
        if not os.path.exists(model_path):
            st.error(f"Model not found at {model_path}")
            return None, None
        
        if not os.path.exists(encoder_path):
            st.error(f"Encoder not found at {encoder_path}")
            return None, None
        
        model = tf.keras.models.load_model(model_path)
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        
        return model, encoder
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def get_audio_info(audio_path):
    """Get audio file information"""
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr
        return {
            'duration': duration,
            'sample_rate': sr,
            'samples': len(audio),
            'file_size': os.path.getsize(audio_path) / 1024  # KB
        }
    except:
        return None

def plot_waveform(audio_path):
    """Plot audio waveform"""
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(audio, sr=sr, ax=ax, color='#2E7D32')
        ax.set_title('Waveform')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        return fig
    except:
        return None

def plot_spectrogram(audio_path):
    """Plot mel-spectrogram"""
    try:
        audio, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE, duration=Config.DURATION)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr,
            n_mels=Config.N_MELS,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            mel_db, sr=sr, 
            hop_length=Config.HOP_LENGTH,
            x_axis='time', y_axis='mel', ax=ax
        )
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Mel-Spectrogram')
        return fig
    except:
        return None

def create_confidence_gauge(confidence, title="Confidence"):
    """Create a gauge chart for confidence"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        title = {'text': title},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#2E7D32"},
            'steps': [
                {'range': [0, 50], 'color': "#ffcccc"},
                {'range': [50, 75], 'color': "#ffffcc"},
                {'range': [75, 100], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
    return fig

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/cricket--v1.png", width=80)
    st.title("🐞 PestSoundCRNN")
    st.markdown("---")
    
    # Model status
    st.subheader("📊 Model Status")
    
    if not st.session_state.model_loaded:
        if st.button("🔌 Load Model", use_container_width=True):
            with st.spinner("Loading model..."):
                model, encoder = load_model_resources()
                if model is not None and encoder is not None:
                    st.session_state.model = model
                    st.session_state.encoder = encoder
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load model")
    else:
        st.success("✅ Model is loaded")
        st.info(f"📈 Model ready for prediction")
        
        if st.button("🔄 Reload Model", use_container_width=True):
            st.session_state.model_loaded = False
            st.session_state.model = None
            st.session_state.encoder = None
            st.rerun()
    
    st.markdown("---")
    
    # Dataset info
    st.subheader("📁 Dataset Info")
    if os.path.exists(Config.DATA_PATH):
        st.info(f"📂 Dataset: {Config.DATASET_FOLDER}")
        
        # Count audio files
        audio_count = 0
        for root, dirs, files in os.walk(Config.DATA_PATH):
            audio_count += len([f for f in files if f.endswith(('.wav', '.mp3'))])
        st.metric("Audio Files", audio_count)
    else:
        st.warning("⚠️ Dataset folder not found")
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    st.slider("Top K predictions", min_value=1, max_value=10, value=5, key="top_k")
    
    # Clear history button
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.prediction_history = []
        st.success("History cleared!")

# ============================================
# MAIN CONTENT
# ============================================
st.markdown('<h1 class="main-header">🐞 Pest Sound Classification</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Deep Learning CRNN Model for Insect Sound Recognition</p>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎵 Single Prediction", 
    "📁 Batch Prediction", 
    "📊 Analytics",
    "ℹ️ Model Info"
])

# ============================================
# TAB 1: SINGLE PREDICTION
# ============================================
with tab1:
    st.header("🎵 Single Audio Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose a WAV file", 
            type=['wav', 'mp3', 'ogg', 'm4a'],
            help="Upload an audio file of pest sound"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = os.path.join("temp_audio.wav")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Display audio info
            audio_info = get_audio_info(temp_path)
            if audio_info:
                st.markdown("**Audio Information:**")
                st.json({
                    "Duration": f"{audio_info['duration']:.2f} seconds",
                    "Sample Rate": f"{audio_info['sample_rate']} Hz",
                    "File Size": f"{audio_info['file_size']:.1f} KB"
                })
            
            # Audio player
            st.audio(temp_path)
    
    with col2:
        if uploaded_file is not None and st.session_state.model_loaded:
            st.subheader("Prediction Results")
            
            with st.spinner("Analyzing audio..."):
                # Extract features and predict
                try:
                    features = extract_features(temp_path)
                    
                    if features is not None:
                        # Make prediction
                        predictions = st.session_state.model.predict(features, verbose=0)[0]
                        
                        # Get top K predictions
                        top_k = st.session_state.top_k
                        top_indices = np.argsort(predictions)[-top_k:][::-1]
                        
                        # Main prediction
                        main_idx = top_indices[0]
                        main_pest = st.session_state.encoder.inverse_transform([main_idx])[0]
                        main_confidence = predictions[main_idx]
                        
                        # Display main prediction in a nice box
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        
                        # Confidence color
                        if main_confidence > 0.8:
                            conf_class = "confidence-high"
                        elif main_confidence > 0.5:
                            conf_class = "confidence-medium"
                        else:
                            conf_class = "confidence-low"
                        
                        st.markdown(f"### 🎯 Primary Prediction")
                        st.markdown(f"#### **{main_pest}**")
                        st.markdown(f"##### Confidence: <span class='{conf_class}'>{main_confidence*100:.2f}%</span>", unsafe_allow_html=True)
                        
                        # Gauge chart
                        gauge_fig = create_confidence_gauge(main_confidence)
                        st.plotly_chart(gauge_fig, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Top K predictions
                        st.subheader(f"📊 Top {top_k} Predictions")
                        
                        # Create dataframe for display
                        pred_data = []
                        for i, idx in enumerate(top_indices, 1):
                            pest = st.session_state.encoder.inverse_transform([idx])[0]
                            conf = predictions[idx]
                            pred_data.append({
                                "Rank": i,
                                "Pest Species": pest,
                                "Confidence": f"{conf*100:.2f}%",
                                "Probability": conf
                            })
                        
                        pred_df = pd.DataFrame(pred_data)
                        
                        # Bar chart
                        fig = px.bar(
                            pred_df, 
                            x="Confidence", 
                            y="Pest Species",
                            orientation='h',
                            color="Confidence",
                            color_continuous_scale=["#ffcccc", "#ffffcc", "#ccffcc"],
                            title="Prediction Probabilities"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display table
                        st.dataframe(pred_df[["Rank", "Pest Species", "Confidence"]], use_container_width=True)
                        
                        # Add to history
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'filename': uploaded_file.name,
                            'prediction': main_pest,
                            'confidence': main_confidence,
                            'top_3': [f"{pest}: {conf:.2f}%" for pest, conf in zip(
                                [st.session_state.encoder.inverse_transform([idx])[0] for idx in top_indices[:3]],
                                [predictions[idx] for idx in top_indices[:3]]
                            )]
                        })
                    else:
                        st.error("Failed to extract features from audio")
                
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        elif uploaded_file is not None and not st.session_state.model_loaded:
            st.warning("⚠️ Please load the model first using the sidebar button")
        
        else:
            st.info("👆 Upload an audio file to get started")

# ============================================
# TAB 2: BATCH PREDICTION
# ============================================
with tab2:
    st.header("📁 Batch Prediction")
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load the model first using the sidebar button")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Multiple Files")
            uploaded_files = st.file_uploader(
                "Choose audio files", 
                type=['wav', 'mp3', 'ogg', 'm4a'],
                accept_multiple_files=True,
                help="Upload multiple audio files for batch prediction"
            )
        
        with col2:
            st.subheader("Folder Path")
            folder_path = st.text_input(
                "Or enter folder path:",
                placeholder="C:/path/to/audio/folder"
            )
            
            if folder_path and os.path.exists(folder_path):
                audio_files = []
                for file in os.listdir(folder_path):
                    if file.endswith(('.wav', '.mp3', '.ogg', '.m4a')):
                        audio_files.append(os.path.join(folder_path, file))
                st.info(f"Found {len(audio_files)} audio files")
                
                if st.button("🔍 Scan Folder"):
                    st.session_state.audio_files = audio_files
        
        # Batch prediction
        files_to_process = []
        
        if uploaded_files:
            files_to_process = uploaded_files
        elif st.session_state.audio_files:
            files_to_process = st.session_state.audio_files
        
        if files_to_process and st.button("🚀 Run Batch Prediction", use_container_width=True):
            st.subheader("Batch Results")
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(files_to_process):
                status_text.text(f"Processing {i+1}/{len(files_to_process)}: {file.name if hasattr(file, 'name') else os.path.basename(file)}")
                
                try:
                    # Save temp file if uploaded
                    if hasattr(file, 'name'):
                        temp_path = f"temp_{i}.wav"
                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())
                        file_path = temp_path
                        filename = file.name
                    else:
                        file_path = file
                        filename = os.path.basename(file)
                    
                    # Predict
                    features = extract_features(file_path)
                    if features is not None:
                        predictions = st.session_state.model.predict(features, verbose=0)[0]
                        top_idx = np.argmax(predictions)
                        pest = st.session_state.encoder.inverse_transform([top_idx])[0]
                        confidence = predictions[top_idx]
                        
                        results.append({
                            'Filename': filename,
                            'Prediction': pest,
                            'Confidence': f"{confidence*100:.2f}%",
                            'Confidence_Value': confidence
                        })
                    
                    # Clean up
                    if hasattr(file, 'name') and os.path.exists(temp_path):
                        os.remove(temp_path)
                
                except Exception as e:
                    results.append({
                        'Filename': filename if 'filename' in locals() else 'Unknown',
                        'Prediction': f"Error: {str(e)}",
                        'Confidence': "0%",
                        'Confidence_Value': 0
                    })
                
                progress_bar.progress((i + 1) / len(files_to_process))
            
            status_text.text("✅ Batch processing complete!")
            
            # Display results
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="batch_predictions.csv",
                mime="text/csv"
            )
            
            # Summary statistics
            st.subheader("Summary Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Files", len(results))
            
            with col2:
                avg_conf = results_df['Confidence_Value'].mean()
                st.metric("Average Confidence", f"{avg_conf*100:.2f}%")
            
            with col3:
                high_conf = len(results_df[results_df['Confidence_Value'] > 0.8])
                st.metric("High Confidence (>80%)", high_conf)

# ============================================
# TAB 3: ANALYTICS
# ============================================
with tab3:
    st.header("📊 Prediction Analytics")
    
    if st.session_state.prediction_history:
        # Create dataframe from history
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predictions", len(history_df))
        
        with col2:
            avg_conf = history_df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_conf*100:.2f}%")
        
        with col3:
            unique_pests = history_df['prediction'].nunique()
            st.metric("Unique Pests Detected", unique_pests)
        
        # Prediction distribution
        st.subheader("Prediction Distribution")
        pest_counts = history_df['prediction'].value_counts()
        
        fig = px.pie(
            values=pest_counts.values,
            names=pest_counts.index,
            title="Pest Types Predicted"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence over time
        st.subheader("Confidence Over Time")
        fig = px.line(
            history_df,
            x=range(len(history_df)),
            y='confidence',
            title="Prediction Confidence Trend",
            labels={'x': 'Prediction #', 'confidence': 'Confidence'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent history table
        st.subheader("Recent Predictions")
        st.dataframe(
            history_df[['timestamp', 'filename', 'prediction', 'confidence']].tail(10),
            use_container_width=True
        )
        
        # Export history
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download History",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )
    
    else:
        st.info("No predictions yet. Make some predictions in the Single Prediction tab!")

# ============================================
# TAB 4: MODEL INFO
# ============================================
with tab4:
    st.header("ℹ️ Model Information")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Model Architecture")
        
        if st.session_state.model_loaded:
            # Get model summary as string
            stringlist = []
            st.session_state.model.summary(print_fn=lambda x: stringlist.append(x))
            model_summary = "\n".join(stringlist)
            
            st.text(model_summary)
            
            # Model parameters
            total_params = st.session_state.model.count_params()
            st.metric("Total Parameters", f"{total_params:,}")
        
        else:
            st.info("Load the model to see architecture")
    
    with col2:
        st.subheader("Configuration")
        
        # Display config in a nice table
        config_data = {
            "Parameter": [
                "Sample Rate",
                "Duration",
                "Mel Bands",
                "FFT Size",
                "Hop Length",
                "Batch Size",
                "Learning Rate",
                "Channels",
                "Augmentation"
            ],
            "Value": [
                f"{Config.SAMPLE_RATE} Hz",
                f"{Config.DURATION} s",
                Config.N_MELS,
                Config.N_FFT,
                Config.HOP_LENGTH,
                Config.BATCH_SIZE,
                Config.LEARNING_RATE,
                getattr(Config, 'N_CHANNELS', 1),
                getattr(Config, 'USE_AUGMENTATION', False)
            ]
        }
        
        config_df = pd.DataFrame(config_data)
        st.dataframe(config_df, use_container_width=True)
        
        st.subheader("Dataset Info")
        if os.path.exists(Config.DATA_PATH):
            dataset_path = os.path.join(Config.DATA_PATH, Config.DATASET_FOLDER)
            if os.path.exists(dataset_path):
                # Count files
                wav_count = 0
                for root, dirs, files in os.walk(dataset_path):
                    wav_count += len([f for f in files if f.endswith('.wav')])
                
                st.metric("Audio Files", wav_count)
                
                # Get classes from encoder if loaded
                if st.session_state.model_loaded:
                    classes = st.session_state.encoder.classes_
                    st.metric("Number of Classes", len(classes))
                    
                    # Show sample classes
                    st.write("**Sample Classes:**")
                    for i, class_name in enumerate(classes[:10]):
                        st.write(f"{i+1}. {class_name}")
                    if len(classes) > 10:
                        st.write(f"... and {len(classes) - 10} more")
            else:
                st.warning("Dataset folder not found")
        else:
            st.warning("Data path not found")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🐞 Pest Sound Classification CRNN Model | Built with TensorFlow & Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Auto-refresh for model loading
if st.session_state.model_loaded and 'model' not in st.session_state:
    st.rerun()