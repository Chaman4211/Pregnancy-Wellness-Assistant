# app.py - Pregnancy Wellness Assistant with Enhanced Features
import streamlit as st
import sqlite3
import hashlib
import datetime
import pandas as pd
import numpy as np
import librosa
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import tempfile
import json
import os
import warnings
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import io
import traceback
import tensorflow as tf
from tensorflow import keras
import re
from collections import Counter

warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Pregnancy Wellness Assistant",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/yourusername/pregnancy-wellness',
        'Report a bug': "https://github.com/yourusername/pregnancy-wellness/issues",
        'About': """
        # Pregnancy Emotional Wellness Assistant
        Supporting maternal mental health through AI-powered voice and text analysis.
        
        **Disclaimer**: This tool provides emotional support only, not medical advice.
        Always consult healthcare providers for medical concerns.
        """
    }
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    /* Main container styling */
    .main-container {
        max-width: 100%;
        padding: 2rem;
    }
    
    /* Auth container styling */
    .auth-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 3rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .auth-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .auth-title {
        color: #FF69B4;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .auth-subtitle {
        color: #666;
        font-size: 1.1rem;
    }
    
    .auth-form {
        margin-top: 2rem;
    }
    
    .auth-input {
        margin-bottom: 1.5rem;
    }
    
    .auth-button {
        width: 100%;
        padding: 0.75rem;
        border-radius: 10px;
        font-size: 1.1rem;
    }
    
    .auth-switch {
        text-align: center;
        margin-top: 1.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid #eee;
    }
    
    /* App styling */
    .main-header {
        font-size: 2.5rem;
        color: #FF69B4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #9370DB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF9800;
        margin: 1rem 0;
    }
    .baby-box {
        background-color: #F3E5F5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #9C27B0;
        margin: 1rem 0;
    }
    .nutrition-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .exercise-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(0,0,0,0.1);
    }
    .emergency-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #F44336;
        margin: 1rem 0;
    }
    .model-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E7D32;
        margin: 1rem 0;
    }
    .audio-wave {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF9800;
        margin: 1rem 0;
    }
    .tab-content {
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATABASE SETUP
# ============================================
conn = sqlite3.connect("pregnancy_wellness.db", check_same_thread=False)
c = conn.cursor()

# Create tables
c.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            email TEXT,
            trimester INTEGER,
            weeks_pregnant INTEGER,
            baby_name TEXT,
            due_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")

c.execute("""CREATE TABLE IF NOT EXISTS emotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            emotion TEXT,
            confidence REAL,
            source TEXT,
            notes TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
            )""")

c.execute("""CREATE TABLE IF NOT EXISTS baby_kicks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            time TEXT,
            kicks INTEGER,
            duration_minutes INTEGER,
            notes TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
            )""")

c.execute("""CREATE TABLE IF NOT EXISTS symptoms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            symptom TEXT,
            severity INTEGER,
            notes TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
            )""")

c.execute("""CREATE TABLE IF NOT EXISTS checkins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            mood TEXT,
            energy INTEGER,
            sleep_hours REAL,
            appetite TEXT,
            notes TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
            )""")

c.execute("""CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            emotion TEXT,
            trimester INTEGER,
            week INTEGER,
            recommendation TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
            )""")

c.execute("""CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            model_type TEXT,
            emotion_detected TEXT,
            confidence REAL,
            feedback TEXT,
            correct_prediction INTEGER DEFAULT 1,
            FOREIGN KEY(user_id) REFERENCES users(id)
            )""")

# Add nutrition and exercise tracking tables
c.execute("""CREATE TABLE IF NOT EXISTS nutrition_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            meal_type TEXT,
            food_items TEXT,
            calories INTEGER,
            nutrients TEXT,
            notes TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
            )""")

c.execute("""CREATE TABLE IF NOT EXISTS exercise_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            exercise_type TEXT,
            duration_minutes INTEGER,
            intensity TEXT,
            calories_burned INTEGER,
            notes TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
            )""")

c.execute("""CREATE TABLE IF NOT EXISTS vitamin_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            vitamin_name TEXT,
            taken BOOLEAN,
            dosage TEXT,
            notes TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
            )""")

conn.commit()

# ============================================
# YOUR RESIDUAL CNN MODEL ARCHITECTURE (Voice)
# ============================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.dropout(out)
        return out

class UltraStrongCNN(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer1 = ResidualBlock(32, 64, stride=2, dropout=0.2)
        self.layer2 = ResidualBlock(64, 128, stride=2, dropout=0.3)
        self.layer3 = ResidualBlock(128, 256, stride=2, dropout=0.4)
        self.layer4 = ResidualBlock(256, 512, stride=2, dropout=0.5)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, n_classes)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.init_conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# ============================================
# TEXT EMOTION DETECTOR (FIXED VERSION)
# ============================================
class TextEmotionDetector:
    def __init__(self):
        """Initialize text emotion detector with correct max length"""
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_length = 300  # Fixed to match your model training
        self.model_loaded = False
        self._load_text_models()
    
    def _load_text_models(self):
        """Load the text-based emotion detection models"""
        try:
            # Load tokenizer
            tokenizer_path = r"C:\Users\Hp\Downloads\PragnancyAI\pregnancy_models\tokenizer.pkl"
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
            
            # Load label encoder
            encoder_path = r"C:\Users\Hp\Downloads\PragnancyAI\pregnancy_models\emotion_encoder.pkl"
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
            
            # Load Keras model
            model_path = r"C:\Users\Hp\Downloads\PragnancyAI\pregnancy_models\best_text_emotion_model_final.keras"
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
                self.model_loaded = True
                
                # Try to infer max_length from model input shape
                try:
                    model_input_shape = self.model.input_shape
                    if model_input_shape is not None and len(model_input_shape) > 1:
                        inferred_max_length = model_input_shape[1]
                        if inferred_max_length is not None:
                            self.max_length = int(inferred_max_length)
                except:
                    pass
                    
            else:
                self.model_loaded = False
                
        except Exception as e:
            self.model_loaded = False
    
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra spaces
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_text(self, text):
        """Analyze text for emotions using deep learning model"""
        if self.model is None or self.tokenizer is None:
            return self._rule_based_analysis(text)
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Tokenize - use the correct max_length
            sequences = self.tokenizer.texts_to_sequences([processed_text])
            
            # Pad sequences with the correct max_length (300)
            padded = keras.preprocessing.sequence.pad_sequences(
                sequences, 
                maxlen=self.max_length, 
                padding='post',
                truncating='post'
            )
            
            # Predict
            predictions = self.model.predict(padded, verbose=0)
            
            # Get top prediction
            emotion_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][emotion_idx])
            
            # Decode emotion
            if self.label_encoder and hasattr(self.label_encoder, 'classes_'):
                if emotion_idx < len(self.label_encoder.classes_):
                    emotion = self.label_encoder.classes_[emotion_idx]
                else:
                    emotion = "Neutral"
            else:
                emotion = "Neutral"
            
            # Get all emotion scores
            emotion_scores = {}
            if self.label_encoder and hasattr(self.label_encoder, 'classes_'):
                for idx, emo in enumerate(self.label_encoder.classes_):
                    if idx < len(predictions[0]):
                        emotion_scores[emo] = float(predictions[0][idx])
            
            return emotion, confidence, emotion_scores, "Text CNN"
            
        except Exception as e:
            return self._rule_based_analysis(text)
    
    def _rule_based_analysis(self, text):
        """Fallback rule-based text analysis"""
        emotion_keywords = {
            'Happy': ['happy', 'joy', 'excited', 'good', 'great', 'wonderful', 'love', 'amazing', 'blessed', 'fantastic'],
            'Calm': ['calm', 'peaceful', 'relaxed', 'content', 'serene', 'peace', 'quiet', 'still', 'tranquil'],
            'Anxious': ['anxious', 'worried', 'nervous', 'stressed', 'tense', 'scared', 'afraid', 'panic', 'uneasy'],
            'Sad': ['sad', 'tired', 'exhausted', 'sleepy', 'fatigued', 'depressed', 'low', 'down', 'blue'],
            'Angry': ['angry', 'mad', 'frustrated', 'irritated', 'annoyed', 'upset', 'furious', 'rage'],
            'Surprised': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'wow'],
            'Neutral': ['okay', 'fine', 'normal', 'alright', 'regular', 'neutral', 'average']
        }
        
        text_lower = text.lower()
        emotion_counts = {emotion: 0 for emotion in emotion_keywords}
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotion_counts[emotion] += 1
        
        total_matches = sum(emotion_counts.values())
        if total_matches > 0:
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
            confidence = min(0.95, dominant_emotion[1] / len(emotion_keywords[dominant_emotion[0]]) * 2)
            
            # Calculate normalized scores
            emotion_scores = {k: v/total_matches for k, v in emotion_counts.items()}
            
            return dominant_emotion[0], confidence, emotion_scores, "Rule-based"
        else:
            # Default to Neutral if no keywords found
            return "Neutral", 0.5, {"Neutral": 1.0}, "Default"

# ============================================
# ENHANCED VOICE EMOTION DETECTOR
# ============================================
class VoiceEmotionDetector:
    def __init__(self):
        """Initialize the voice emotion detector"""
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emotion_classes = ['Anxious', 'Calm', 'Frustrated', 'Happy', 'Surprised', 'Tired/Sad', 'Uncomfortable']
        self.model_weights_loaded = False
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all models with better error handling"""
        # Always create label encoder first
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.emotion_classes)
        
        # Try to load pre-trained components
        self._load_pretrained_components()
        
        # Always create model architecture
        self._create_model()
    
    def _load_pretrained_components(self):
        """Try to load pre-trained encoder and scaler"""
        try:
            # Try to load encoder
            encoder_path = r"C:\Users\Hp\Downloads\PragnancyAI\pregnancy_models\pregnancy_encoder.pkl"
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
            
            # Try to load scaler
            scaler_path = r"C:\Users\Hp\Downloads\PragnancyAI\pregnancy_models\pregnancy_scaler.pkl"
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                    
        except Exception as e:
            # Use default encoder if loading fails
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.emotion_classes)
    
    def _create_model(self):
        """Create and load the PyTorch model"""
        model_path = r"C:\Users\Hp\Downloads\PragnancyAI\pregnancy_models\best_emotion_cnn.pth"
        
        # Create model with correct number of classes
        n_classes = len(self.label_encoder.classes_)
        self.model = UltraStrongCNN(n_classes=n_classes)
        self.model.to(self.device)
        
        # Try to load weights if file exists
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                state_dict = None
                if isinstance(checkpoint, dict):
                    possible_keys = ['model_state_dict', 'state_dict', 'model']
                    for key in possible_keys:
                        if key in checkpoint:
                            state_dict = checkpoint[key]
                            break
                
                if state_dict is None and not isinstance(checkpoint, dict):
                    state_dict = checkpoint
                
                if state_dict is not None:
                    model_state_dict = self.model.state_dict()
                    filtered_state_dict = {}
                    for key in state_dict.keys():
                        if key in model_state_dict:
                            if state_dict[key].shape == model_state_dict[key].shape:
                                filtered_state_dict[key] = state_dict[key]
                    
                    if filtered_state_dict:
                        self.model.load_state_dict(filtered_state_dict, strict=False)
                        self.model_weights_loaded = True
                
            except Exception as e:
                pass
        
        self.model.eval()
    
    def extract_features(self, audio_path, max_len=128):
        """Extract comprehensive audio features"""
        try:
            y, sr = librosa.load(audio_path, sr=16000, duration=3)
            y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
            
            n_mfcc = 40
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
            
            if mfcc.shape[1] < max_len:
                pad_width = max_len - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            else:
                mfcc = mfcc[:, :max_len]
            
            if self.scaler is not None:
                original_shape = mfcc.shape
                mfcc_flat = mfcc.flatten().reshape(1, -1)
                mfcc_flat = self.scaler.transform(mfcc_flat)
                mfcc = mfcc_flat.reshape(original_shape)
            
            features = np.expand_dims(mfcc, axis=0)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            basic_features = {
                'energy': np.mean(librosa.feature.rms(y=y)),
                'pitch': np.mean(librosa.yin(y, fmin=50, fmax=500)),
                'zcr': np.mean(librosa.feature.zero_crossing_rate(y)),
                'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
                'duration': len(y) / sr
            }
            
            return features_tensor, basic_features, (y, sr)
            
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            return None, None, None
    
    def predict_emotion(self, audio_path):
        """Predict emotion using the trained model"""
        try:
            features_tensor, basic_features, audio_data = self.extract_features(audio_path)
            
            if features_tensor is None:
                return self._enhanced_fallback_prediction(basic_features)
            
            if self.model is not None and self.model_weights_loaded:
                features_tensor = features_tensor.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(features_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    
                    confidence = confidence.item()
                    predicted_idx = predicted_idx.item()
                    
                    if predicted_idx < len(self.label_encoder.classes_):
                        emotion = self.label_encoder.classes_[predicted_idx]
                    else:
                        emotion = 'Calm'
                    
                    all_probs = probabilities.squeeze().cpu().numpy()
                    emotion_scores = {}
                    for idx, emo in enumerate(self.label_encoder.classes_):
                        if idx < len(all_probs):
                            emotion_scores[emo] = all_probs[idx]
                        else:
                            emotion_scores[emo] = 0.0
                    
                    total = sum(emotion_scores.values())
                    if total > 0:
                        normalized_scores = {k: v/total for k, v in emotion_scores.items()}
                    else:
                        normalized_scores = {k: 1.0/len(emotion_scores) for k in emotion_scores.keys()}
                    
                    return emotion, confidence, normalized_scores, emotion, "Trained CNN"
            else:
                return self._enhanced_fallback_prediction(basic_features)
                
        except Exception as e:
            st.warning(f"Prediction error: {e}")
            return self._basic_fallback_prediction()
    
    def _enhanced_fallback_prediction(self, basic_features):
        """Enhanced fallback prediction using audio features"""
        if basic_features is None:
            return self._basic_fallback_prediction()
        
        emotion_scores = {emotion: 0.1 for emotion in self.emotion_classes}
        
        if 'energy' in basic_features:
            if basic_features['energy'] < 0.03:
                emotion_scores['Tired/Sad'] += 0.4
                emotion_scores['Calm'] += 0.2
            elif basic_features['energy'] > 0.1:
                emotion_scores['Happy'] += 0.3
                emotion_scores['Surprised'] += 0.2
        
        if 'pitch' in basic_features:
            if basic_features['pitch'] > 200:
                emotion_scores['Anxious'] += 0.3
                emotion_scores['Surprised'] += 0.2
            elif basic_features['pitch'] < 100:
                emotion_scores['Calm'] += 0.3
                emotion_scores['Tired/Sad'] += 0.2
        
        total = sum(emotion_scores.values())
        normalized_scores = {k: v/total for k, v in emotion_scores.items()}
        dominant_emotion = max(normalized_scores.items(), key=lambda x: x[1])
        confidence = min(0.85, 0.5 + dominant_emotion[1])
        
        return dominant_emotion[0], confidence, normalized_scores, 'audio_features', 'Enhanced Audio Analysis'
    
    def _basic_fallback_prediction(self):
        """Basic fallback prediction"""
        emotion_scores = {
            'Calm': 0.35,
            'Happy': 0.25,
            'Tired/Sad': 0.15,
            'Anxious': 0.1,
            'Surprised': 0.08,
            'Frustrated': 0.05,
            'Uncomfortable': 0.02
        }
        
        total = sum(emotion_scores.values())
        normalized_scores = {k: v/total for k, v in emotion_scores.items()}
        
        return 'Calm', 0.5, normalized_scores, 'basic_fallback', 'Basic Analysis'

# ============================================
# INITIALIZE DETECTORS
# ============================================
text_detector = TextEmotionDetector()
voice_detector = VoiceEmotionDetector()

# ============================================
# PREGNANCY INFO
# ============================================
PREGNANCY_TIMELINE = {
    1: {"desc": "Conception occurs\nBlastocyst implants in uterus\nPlacenta begins to form", "size": "Poppy seed"},
    2: {"desc": "Baby's brain, spinal cord, heart begin", "size": "Apple seed"},
    3: {"desc": "Heart beats, limb buds appear", "size": "Lentil"},
    4: {"desc": "Brain growing rapidly\nEyes, ears, mouth forming", "size": "Blueberry"},
    5: {"desc": "All major organs forming\nWebbed fingers and toes", "size": "Raspberry"},
    6: {"desc": "Embryo becomes fetus\nTiny muscles can move", "size": "Grape"},
    7: {"desc": "Organs fully formed, beginning to function\nFingernails and hair forming", "size": "Kumquat"},
    8: {"desc": "Baby kicking and stretching\nGenitals developing", "size": "Fig"},
    9: {"desc": "Reflexes developing\nCan open and close fingers", "size": "Lime"},
    10: {"desc": "Vocal cords developing\nCan suck thumb", "size": "Lemon"},
    11: {"desc": "Facial expressions possible\nFine hair (lanugo) appears", "size": "Peach"},
    12: {"desc": "Can sense light\nTaste buds forming", "size": "Apple"},
    13: {"desc": "Hearing developing\nSex identifiable on ultrasound", "size": "Avocado"},
    14: {"desc": "Fat stores developing\nSweat glands forming", "size": "Turnip"},
    15: {"desc": "Yawning and hiccupping\nCan hear mom's heartbeat", "size": "Bell pepper"},
    16: {"desc": "Protective vernix coating skin\nHair growing on scalp", "size": "Heirloom tomato"},
    17: {"desc": "Midpoint of pregnancy\nMom may feel movement (quickening)", "size": "Banana"},
    18: {"desc": "Regular sleep/wake cycles\nTaste of amniotic fluid", "size": "Carrot"},
    19: {"desc": "Eyebrows and eyelashes visible\nFingerprints forming", "size": "Spaghetti squash"},
    20: {"desc": "Rapid eye movements\nLoud noises may startle baby", "size": "Grapefruit"},
    21: {"desc": "Viability milestone (can survive with NICU care)\nLungs developing", "size": "Ear of corn"},
    22: {"desc": "Responds to familiar voices\nHand dominance may show", "size": "Rutabaga"},
    23: {"desc": "Eyes begin to open\nBreathing movements (practice)", "size": "Scallion bunch"},
    24: {"desc": "Brain developing rapidly\nRecognizes mom's voice", "size": "Cauliflower"},
    25: {"desc": "Can blink eyes\nDreaming may begin (REM sleep)", "size": "Eggplant"},
    26: {"desc": "Kicking and punching vigorously\nBones fully developed", "size": "Butternut squash"},
    27: {"desc": "Controls body temperature\nRed blood cell production begins", "size": "Large cabbage"},
    28: {"desc": "All five senses functional\nGains weight rapidly", "size": "Coconut"},
    29: {"desc": "Toenails visible\nLess room to move", "size": "Jicama"},
    30: {"desc": "Immune system developing\nBones hardening (except skull)", "size": "Pineapple"},
    31: {"desc": "Lungs nearly mature\nFingernails reach fingertips", "size": "Cantaloupe"},
    32: {"desc": "Most development complete\nGaining fat for temperature regulation", "size": "Honeydew melon"},
    33: {"desc": "May descend into pelvis (engagement)\nSkin smoothing out", "size": "Head of romaine lettuce"},
    34: {"desc": "Early term\nPractice breathing continues\nSucking reflex strong", "size": "Swiss chard bunch"},
    35: {"desc": "Brain continues developing\nFirm grasp", "size": "Leek"},
    36: {"desc": "Full term\nShedding vernix coating\nReady for birth", "size": "Mini watermelon"},
    37: {"desc": "Due date week\nBaby's organs ready for outside world", "size": "Small pumpkin"},
    38: {"desc": "Average newborn: 7.5 lbs, 20 inches", "size": "Small pumpkin"},
    39: {"desc": "Ready for delivery", "size": "Small pumpkin"},
    40: {"desc": "Full term achieved\nReady for delivery", "size": "Small pumpkin"},
}

def get_baby_info_by_week(week):
    """Get baby development information for specific week"""
    week = max(1, min(40, week))
    info = PREGNANCY_TIMELINE.get(week, {"desc": "Information not available", "size": "N/A"})
    return f"**Week {week}**\n- {info['desc']}\n- **Size:** {info['size']}"

def get_trimester(week):
    """Get trimester based on week"""
    if week <= 13:
        return 1
    elif week <= 27:
        return 2
    else:
        return 3

# ============================================
# DATABASE HELPER FUNCTIONS
# ============================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def signup_user(username, password, email, trimester=1, weeks=1, baby_name="Little One"):
    try:
        c.execute("""INSERT INTO users 
                    (username, password, email, trimester, weeks_pregnant, baby_name) 
                    VALUES (?, ?, ?, ?, ?, ?)""",
                  (username, hash_password(password), email, trimester, weeks, baby_name))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", 
              (username, hash_password(password)))
    return c.fetchone()

def update_user_profile(user_id, trimester, weeks_pregnant, baby_name, due_date=None):
    c.execute("""UPDATE users SET 
                trimester=?, weeks_pregnant=?, baby_name=?, due_date=?
                WHERE id=?""",
              (trimester, weeks_pregnant, baby_name, due_date, user_id))
    conn.commit()

def add_emotion(user_id, emotion, confidence=0.0, source="voice", notes=""):
    c.execute("""INSERT INTO emotions 
                (user_id, date, emotion, confidence, source, notes)
                VALUES (?, ?, ?, ?, ?, ?)""",
              (user_id, str(datetime.date.today()), emotion, confidence, source, notes))
    conn.commit()

def add_baby_kick(user_id, kicks, duration_minutes=10, notes=""):
    c.execute("""INSERT INTO baby_kicks 
                (user_id, date, time, kicks, duration_minutes, notes)
                VALUES (?, ?, ?, ?, ?, ?)""",
              (user_id, str(datetime.date.today()), 
               datetime.datetime.now().strftime("%H:%M"), 
               kicks, duration_minutes, notes))
    conn.commit()

def add_symptom(user_id, symptom, severity=5, notes=""):
    c.execute("""INSERT INTO symptoms 
                (user_id, date, symptom, severity, notes)
                VALUES (?, ?, ?, ?, ?)""",
              (user_id, str(datetime.date.today()), symptom, severity, notes))
    conn.commit()

def add_daily_checkin(user_id, mood, energy, sleep_hours=8.0, appetite="Normal", notes=""):
    c.execute("""INSERT INTO checkins 
                (user_id, date, mood, energy, sleep_hours, appetite, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
              (user_id, str(datetime.date.today()), mood, energy, sleep_hours, appetite, notes))
    conn.commit()

def add_recommendation(user_id, emotion, trimester, week, recommendation):
    c.execute("""INSERT INTO recommendations 
                (user_id, date, emotion, trimester, week, recommendation)
                VALUES (?, ?, ?, ?, ?, ?)""",
              (user_id, str(datetime.date.today()), emotion, trimester, week, recommendation))
    conn.commit()

def get_user_data(user_id):
    c.execute("SELECT * FROM users WHERE id=?", (user_id,))
    return c.fetchone()

def get_emotion_history(user_id, days=30):
    c.execute("""SELECT date, emotion, confidence, source, notes 
                FROM emotions 
                WHERE user_id=? AND date >= date('now', ?)
                ORDER BY date DESC""",
              (user_id, f'-{days} days'))
    return pd.DataFrame(c.fetchall(), columns=["date", "emotion", "confidence", "source", "notes"])

def get_baby_kicks_history(user_id, days=30):
    c.execute("""SELECT date, time, kicks, duration_minutes, notes 
                FROM baby_kicks 
                WHERE user_id=? AND date >= date('now', ?)
                ORDER BY date DESC, time DESC""",
              (user_id, f'-{days} days'))
    return pd.DataFrame(c.fetchall(), columns=["date", "time", "kicks", "duration_minutes", "notes"])

def get_symptoms_history(user_id, days=30):
    c.execute("""SELECT date, symptom, severity, notes 
                FROM symptoms 
                WHERE user_id=? AND date >= date('now', ?)
                ORDER BY date DESC""",
              (user_id, f'-{days} days'))
    return pd.DataFrame(c.fetchall(), columns=["date", "symptom", "severity", "notes"])

def get_checkins_history(user_id, days=30):
    c.execute("""SELECT date, mood, energy, sleep_hours, appetite, notes 
                FROM checkins 
                WHERE user_id=? AND date >= date('now', ?)
                ORDER BY date DESC""",
              (user_id, f'-{days} days'))
    return pd.DataFrame(c.fetchall(), columns=["date", "mood", "energy", "sleep_hours", "appetite", "notes"])

# ============================================
# NUTRITION & EXERCISE DATABASE FUNCTIONS
# ============================================
def add_nutrition_log(user_id, meal_type, food_items, calories, nutrients="", notes=""):
    c.execute("""INSERT INTO nutrition_logs 
                (user_id, date, meal_type, food_items, calories, nutrients, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
              (user_id, str(datetime.date.today()), meal_type, food_items, calories, nutrients, notes))
    conn.commit()

def add_exercise_log(user_id, exercise_type, duration_minutes, intensity, calories_burned=0, notes=""):
    c.execute("""INSERT INTO exercise_logs 
                (user_id, date, exercise_type, duration_minutes, intensity, calories_burned, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
              (user_id, str(datetime.date.today()), exercise_type, duration_minutes, intensity, calories_burned, notes))
    conn.commit()

def add_vitamin_log(user_id, vitamin_name, taken=True, dosage="", notes=""):
    c.execute("""INSERT INTO vitamin_logs 
                (user_id, date, vitamin_name, taken, dosage, notes)
                VALUES (?, ?, ?, ?, ?, ?)""",
              (user_id, str(datetime.date.today()), vitamin_name, taken, dosage, notes))
    conn.commit()

def get_nutrition_history(user_id, days=7):
    c.execute("""SELECT date, meal_type, food_items, calories, nutrients, notes 
                FROM nutrition_logs 
                WHERE user_id=? AND date >= date('now', ?)
                ORDER BY date DESC""",
              (user_id, f'-{days} days'))
    return pd.DataFrame(c.fetchall(), columns=["date", "meal_type", "food_items", "calories", "nutrients", "notes"])

def get_exercise_history(user_id, days=7):
    c.execute("""SELECT date, exercise_type, duration_minutes, intensity, calories_burned, notes 
                FROM exercise_logs 
                WHERE user_id=? AND date >= date('now', ?)
                ORDER BY date DESC""",
              (user_id, f'-{days} days'))
    return pd.DataFrame(c.fetchall(), columns=["date", "exercise_type", "duration_minutes", "intensity", "calories_burned", "notes"])

def get_vitamin_history(user_id, days=7):
    c.execute("""SELECT date, vitamin_name, taken, dosage, notes 
                FROM vitamin_logs 
                WHERE user_id=? AND date >= date('now', ?)
                ORDER BY date DESC""",
              (user_id, f'-{days} days'))
    return pd.DataFrame(c.fetchall(), columns=["date", "vitamin_name", "taken", "dosage", "notes"])

# ============================================
# NUTRITION & EXERCISE RECOMMENDATIONS
# ============================================
def get_nutrition_recommendations(trimester, week):
    """Get personalized nutrition recommendations"""
    nutrition_data = {
        1: {
            "essential": [
                "Folic Acid: 400-800 mcg daily (crucial for neural tube development)",
                "Iron: 27 mg daily (supports increased blood volume)",
                "Calcium: 1000 mg daily (for baby's bone development)",
                "Vitamin D: 600 IU daily (helps absorb calcium)",
                "Protein: 75-100g daily (for tissue growth)"
            ],
            "foods": [
                "🍊 Citrus fruits for Vitamin C & folate",
                "🥦 Dark leafy greens for iron & folate",
                "🥜 Nuts and seeds for healthy fats",
                "🥚 Eggs for protein and choline",
                "🥛 Dairy for calcium and vitamin D"
            ],
            "tips": [
                "Eat small, frequent meals to combat nausea",
                "Stay hydrated with 8-10 glasses of water",
                "Avoid raw fish, unpasteurized dairy",
                "Limit caffeine to 200mg per day"
            ]
        },
        2: {
            "essential": [
                "Iron: 27 mg daily (continues to be crucial)",
                "Calcium: 1000 mg daily",
                "Omega-3 DHA: 200-300 mg daily (for brain development)",
                "Vitamin C: 85 mg daily (helps iron absorption)",
                "Zinc: 11 mg daily (supports immune system)"
            ],
            "foods": [
                "🐟 Salmon for Omega-3 (2 servings/week)",
                "🥩 Lean meats for iron",
                "🍠 Sweet potatoes for Vitamin A",
                "🌾 Whole grains for fiber and B vitamins",
                "🥑 Avocados for healthy fats"
            ],
            "tips": [
                "Increase calorie intake by 300-350 calories",
                "Focus on iron-rich foods",
                "Include fiber to prevent constipation",
                "Continue prenatal vitamins"
            ]
        },
        3: {
            "essential": [
                "Iron: 27 mg daily (prepares for blood loss during delivery)",
                "Calcium: 1000 mg daily",
                "Vitamin K: 90 mcg daily (for blood clotting)",
                "Protein: 100g+ daily (for final growth spurt)",
                "Magnesium: 350-400 mg daily (prevents cramps)"
            ],
            "foods": [
                "🍳 Lean proteins for energy",
                "🥬 Leafy greens for iron and vitamin K",
                "🍌 Bananas for potassium and magnesium",
                "💧 Water-rich fruits for hydration",
                "🌰 Almonds for magnesium and protein"
            ],
            "tips": [
                "Eat smaller, more frequent meals",
                "Stay hydrated to prevent contractions",
                "Focus on nutrient-dense foods",
                "Avoid heavy, spicy meals before bed"
            ]
        }
    }
    
    trimester_info = nutrition_data.get(trimester, nutrition_data[1])
    
    week_specific = {
        6: ["Ginger tea for nausea", "Crackers before getting out of bed"],
        12: ["Increase protein intake", "Start focusing on iron-rich foods"],
        20: ["Increase Omega-3 intake", "Focus on calcium for baby's bones"],
        28: ["Increase fiber intake", "Monitor protein consumption"],
        36: ["Smaller, more frequent meals", "Stay well-hydrated"]
    }
    
    week_tips = week_specific.get(week, [])
    
    return trimester_info, week_tips

def get_exercise_recommendations(trimester, week):
    """Get personalized exercise recommendations"""
    exercise_data = {
        1: {
            "safe": [
                "🚶 Walking: 30 minutes daily",
                "🧘 Prenatal yoga: 20-30 minutes",
                "🏊 Swimming: Gentle laps or water aerobics",
                "💃 Low-impact aerobics: Modified routines",
                "🤸 Stretching: Gentle daily stretches"
            ],
            "avoid": [
                "High-impact sports",
                "Contact sports",
                "Exercises on back after 16 weeks",
                "Scuba diving",
                "Hot yoga or hot pilates"
            ],
            "benefits": [
                "Reduces fatigue and nausea",
                "Improves mood and sleep",
                "Prepares body for pregnancy changes",
                "Maintains healthy weight gain"
            ]
        },
        2: {
            "safe": [
                "🚶 Walking: Continue daily",
                "🧘 Prenatal yoga: Focus on balance",
                "🏋️ Light strength training: With modifications",
                "🚴 Stationary cycling: Low resistance",
                "💪 Pelvic floor exercises: Daily Kegels"
            ],
            "avoid": [
                "Exercises lying flat on back",
                "Heavy weight lifting",
                "Activities with risk of falling",
                "High-altitude training",
                "Exercises that cause pain"
            ],
            "benefits": [
                "Reduces back pain",
                "Improves circulation",
                "Helps with posture",
                "Prepares for labor"
            ]
        },
        3: {
            "safe": [
                "🚶 Walking: Shorter, more frequent walks",
                "🧘 Prenatal yoga: Focus on breathing",
                "💪 Pelvic tilts and exercises",
                "🤰 Birth ball exercises",
                "💃 Slow dancing or gentle movement"
            ],
            "avoid": [
                "Exercises requiring balance",
                "High-intensity workouts",
                "Exercises on back",
                "Jumping or bouncing movements",
                "Any exercise causing discomfort"
            ],
            "benefits": [
                "Eases labor and delivery",
                "Reduces swelling",
                "Improves sleep",
                "Maintains strength for postpartum"
            ]
        }
    }
    
    trimester_info = exercise_data.get(trimester, exercise_data[1])
    
    week_specific = {
        12: ["Start pelvic floor exercises", "Establish consistent routine"],
        20: ["Modify exercises as belly grows", "Focus on posture"],
        28: ["Reduce intensity if needed", "Listen to body signals"],
        36: ["Focus on breathing exercises", "Gentle movement only"]
    }
    
    week_tips = week_specific.get(week, [])
    
    return trimester_info, week_tips

def get_vitamin_recommendations(trimester):
    """Get vitamin and supplement recommendations"""
    vitamins = {
        "essential": [
            "💊 Prenatal Multivitamin: Daily",
            "💊 Folic Acid: 400-800 mcg daily",
            "💊 Iron: 27 mg daily (as needed)",
            "💊 Calcium + Vitamin D: 1000 mg + 600 IU",
            "💊 Omega-3 DHA: 200-300 mg daily"
        ],
        "optional": [
            "🌿 Ginger: For nausea (as tea or supplement)",
            "🌿 Probiotics: For digestive health",
            "🌿 Magnesium: For leg cramps (consult doctor)",
            "🌿 Vitamin B6: For nausea relief"
        ],
        "tips": [
            "Take prenatal vitamins with food",
            "Iron is best absorbed with Vitamin C",
            "Calcium can interfere with iron absorption",
            "Always consult doctor before adding supplements"
        ]
    }
    
    trimester_specific = {
        1: ["Focus on folic acid", "Start prenatal vitamins early"],
        2: ["Continue all essentials", "Consider Omega-3 for brain development"],
        3: ["Maintain all vitamins", "Extra iron may be needed"]
    }
    
    extra_tips = trimester_specific.get(trimester, [])
    
    return vitamins, extra_tips

# ============================================
# EMOTIONAL RECOMMENDATIONS
# ============================================
def get_emotional_recommendations(emotion, trimester, week):
    """Get personalized emotional recommendations"""
    recommendations = {
        'Anxious': [
            "Practice deep breathing: 4 seconds in, 7 hold, 8 out",
            "Write down your worries in a pregnancy journal",
            "Talk to your partner about how you're feeling",
            "Listen to calming music",
            "Try prenatal yoga"
        ],
        'Calm': [
            "Enjoy this peaceful moment",
            "Practice gratitude journaling",
            "Share your calm feelings with your partner",
            "Take a gentle walk in nature",
            "Meditate for 10 minutes"
        ],
        'Frustrated': [
            "Take a few deep breaths and count to 10",
            "Express your feelings in a journal",
            "Try gentle stretching exercises",
            "Listen to soothing music",
            "Talk to someone who understands"
        ],
        'Happy': [
            "Capture this happy moment in your pregnancy journal",
            "Share the joy with loved ones",
            "Do something special to celebrate",
            "Take prenatal photos",
            "Play your favorite music"
        ],
        'Surprised': [
            "Embrace the unexpected feelings",
            "Share your surprise with someone close",
            "Take a moment to process your emotions",
            "Write about what surprised you",
            "Enjoy the spontaneity of pregnancy"
        ],
        'Tired/Sad': [
            "Rest when you can - your body is doing important work",
            "Drink plenty of water and have a healthy snack",
            "Take a short nap if possible",
            "Avoid heavy chores",
            "Practice light stretching"
        ],
        'Uncomfortable': [
            "Change positions frequently",
            "Use pillows for support when sitting or lying down",
            "Take a warm bath (not hot)",
            "Practice gentle movements",
            "Wear loose, comfortable clothing"
        ],
        'Sad': [
            "Allow yourself to feel these emotions - they're valid",
            "Reach out to a loved one or support group",
            "Gentle movement can help boost mood",
            "Consider speaking with a counselor specializing in prenatal care",
            "Remember this is temporary and help is available"
        ],
        'Angry': [
            "Step away from the situation temporarily",
            "Practice deep breathing exercises",
            "Express your feelings in a safe, constructive way",
            "Gentle physical activity can help release tension",
            "Talk to someone neutral about what's bothering you"
        ],
        'Neutral': [
            "Practice mindfulness to stay present",
            "Use this balanced state to plan and organize",
            "Check in with your body's needs",
            "Gentle stretching or walking",
            "Enjoy the stability of this moment"
        ]
    }
    
    trimester_tips = {
        1: [
            "Focus on hydration",
            "Eat small, frequent meals",
            "Rest often",
            "Avoid stress triggers",
            "Take prenatal vitamins regularly"
        ],
        2: [
            "Enjoy your increased energy",
            "Start prenatal yoga",
            "Begin planning for baby",
            "Track fetal movements",
            "Attend all prenatal check-ups"
        ],
        3: [
            "Practice relaxation techniques",
            "Prepare your hospital bag",
            "Rest frequently",
            "Practice breathing exercises",
            "Discuss birth plan with your provider"
        ]
    }
    
    week_specific_tips = {
        1: ["Congratulations on your pregnancy! Start taking prenatal vitamins", "Schedule your first prenatal appointment"],
        4: ["Baby's neural tube is forming - ensure adequate folic acid intake", "Avoid alcohol and smoking"],
        8: ["All major organs are forming - eat nutritious foods", "Rest as much as possible"],
        12: ["First trimester almost complete! Nausea may start easing", "Consider sharing the news with family"],
        20: ["Midpoint of pregnancy! Monitor baby movements regularly", "Schedule your anatomy scan"],
        24: ["Start counting kicks daily", "Consider taking childbirth classes"],
        28: ["Third trimester begins! Rest more often", "Monitor blood pressure regularly"],
        32: ["Practice perineal massage", "Pack your hospital bag"],
        36: ["Check hospital route and timing", "Practice relaxation techniques daily"],
        40: ["Stay calm and wait for labor signs", "Rest and conserve energy"]
    }
    
    base_recs = recommendations.get(emotion, ["Be kind to yourself today"])
    trimester_recs = trimester_tips.get(trimester, [])
    week_recs = week_specific_tips.get(week, [])
    
    return base_recs[:3] + trimester_recs[:2] + week_recs[:2]

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================
def plot_audio_waveform(y, sr):
    """Plot audio waveform"""
    fig = go.Figure()
    
    # Create time axis
    times = np.linspace(0, len(y)/sr, len(y))
    
    fig.add_trace(go.Scatter(
        x=times,
        y=y,
        mode='lines',
        name='Audio Waveform',
        line=dict(color='#FF69B4', width=1),
        fill='tozeroy',
        fillcolor='rgba(255, 105, 180, 0.2)'
    ))
    
    fig.update_layout(
        title="Audio Waveform",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=200,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def plot_emotion_radar(scores):
    """Plot emotion scores as radar chart"""
    fig = go.Figure()
    
    categories = list(scores.keys())
    values = list(scores.values())
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Emotion Scores',
        line=dict(color='#9370DB', width=2),
        fillcolor='rgba(147, 112, 219, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat='.0%'
            )
        ),
        showlegend=False,
        height=300,
        margin=dict(l=50, r=50, t=30, b=30)
    )
    
    return fig

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if 'login' not in st.session_state:
    st.session_state.login = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = None
if 'pregnancy_week' not in st.session_state:
    st.session_state.pregnancy_week = 1
if 'trimester' not in st.session_state:
    st.session_state.trimester = 1
if 'page' not in st.session_state:
    st.session_state.page = "login"
if 'model_feedback' not in st.session_state:
    st.session_state.model_feedback = None
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Dashboard"

# ============================================
# AUTHENTICATION PAGES
# ============================================
def show_login_page():
    """Display login page"""
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="auth-header">', unsafe_allow_html=True)
    st.markdown('<h1 class="auth-title">🤰 Pregnancy Wellness</h1>', unsafe_allow_html=True)
    st.markdown('<p class="auth-subtitle">Login to your account</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Login form
    st.markdown('<div class="auth-form">', unsafe_allow_html=True)
    
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🔐 Login", use_container_width=True):
            if username and password:
                user = login_user(username, password)
                if user:
                    st.session_state.login = True
                    st.session_state.user = {
                        "id": user[0],
                        "username": user[1],
                        "email": user[3],
                        "trimester": user[4],
                        "weeks_pregnant": user[5],
                        "baby_name": user[6]
                    }
                    st.session_state.pregnancy_week = user[5]
                    st.session_state.trimester = user[4]
                    st.session_state.page = "main"
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            else:
                st.error("Please fill all fields")
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Switch to signup
    st.markdown('<div class="auth-switch">', unsafe_allow_html=True)
    st.write("Don't have an account?")
    if st.button("Create Account", use_container_width=True):
        st.session_state.page = "signup"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_signup_page():
    """Display signup page"""
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="auth-header">', unsafe_allow_html=True)
    st.markdown('<h1 class="auth-title">🤰 Create Account</h1>', unsafe_allow_html=True)
    st.markdown('<p class="auth-subtitle">Join Pregnancy Wellness Assistant</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Signup form
    st.markdown('<div class="auth-form">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("Username", key="signup_username")
        email = st.text_input("Email", key="signup_email")
        baby_name = st.text_input("Baby's Name (optional)", value="Little One", key="signup_baby")
    
    with col2:
        password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
        trimester = st.selectbox("Trimester", [1, 2, 3], index=0, key="signup_trimester")
        weeks = st.number_input("Weeks Pregnant", min_value=1, max_value=42, value=1, key="signup_weeks")
    
    # Auto-update trimester based on weeks
    if weeks <= 13:
        trimester = 1
    elif weeks <= 27:
        trimester = 2
    else:
        trimester = 3
    
    st.info(f"**Selected:** Week {weeks} | Trimester {trimester}")
    
    # Terms and conditions
    agree = st.checkbox("I agree to the terms and conditions", key="signup_agree")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("📝 Create Account", use_container_width=True, type="primary"):
            if not all([username, password, email]):
                st.error("Please fill all required fields")
            elif password != confirm_password:
                st.error("Passwords do not match")
            elif not agree:
                st.error("Please agree to terms and conditions")
            else:
                if signup_user(username, password, email, trimester, weeks, baby_name):
                    st.success("Account created successfully!")
                    st.info("Please login with your credentials")
                    st.session_state.page = "login"
                    st.rerun()
                else:
                    st.error("Username already exists. Please choose another.")
    
    with col2:
        if st.button("🔙 Back to Login", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Terms and conditions expander
    with st.expander("📋 Terms and Conditions"):
        st.write("""
        **Pregnancy Wellness Assistant - Terms of Use**
        
        1. **Purpose**: This application provides emotional support and wellness tracking for pregnant individuals.
        
        2. **Medical Disclaimer**: This is NOT a medical tool. Always consult healthcare professionals for medical advice.
        
        3. **Data Privacy**: All data is stored locally on your device. We do not collect or share personal information.
        
        4. **User Responsibility**: You are responsible for maintaining the confidentiality of your login credentials.
        
        5. **Limitation of Liability**: The app developers are not liable for any decisions made based on app recommendations.
        
        6. **Acceptance**: By creating an account, you agree to these terms.
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# ENHANCED MAIN APP WITH NEW TABS
# ============================================
def show_main_app():
    """Display main application after login"""
    if not st.session_state.login:
        st.session_state.page = "login"
        st.rerun()
    
    user_id = st.session_state.user["id"]
    username = st.session_state.user["username"]
    trimester = st.session_state.user["trimester"]
    weeks_pregnant = st.session_state.user["weeks_pregnant"]
    baby_name = st.session_state.user["baby_name"]
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; 
                    padding: 1rem; 
                    border-radius: 10px;
                    margin-bottom: 1rem;">
        <h3 style="margin:0;">Welcome, {username}!</h3>
        <p style="margin:0; font-size:0.9rem;">🤰 {baby_name} - Week {weeks_pregnant}</p>
        <p style="margin:0; font-size:0.8rem;">Trimester {trimester}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Profile Management
        st.markdown("### 👤 Your Profile")
        with st.expander("Update Pregnancy Details"):
            new_weeks = st.slider("Weeks Pregnant", 1, 42, weeks_pregnant, key="update_weeks")
            # Auto-calculate trimester based on weeks
            if new_weeks <= 13:
                new_trimester = 1
            elif new_weeks <= 27:
                new_trimester = 2
            else:
                new_trimester = 3
            
            new_baby_name = st.text_input("Baby's Name", value=baby_name, key="update_baby")
            due_date = st.date_input("Due Date (optional)", key="update_due_date")
            
            st.info(f"Trimester will automatically update to: {new_trimester}")
            
            if st.button("Update Profile", use_container_width=True, key="update_button"):
                update_user_profile(user_id, new_trimester, new_weeks, new_baby_name, 
                                  str(due_date) if due_date else None)
                st.session_state.user["trimester"] = new_trimester
                st.session_state.user["weeks_pregnant"] = new_weeks
                st.session_state.user["baby_name"] = new_baby_name
                st.session_state.pregnancy_week = new_weeks
                st.session_state.trimester = new_trimester
                st.success("Profile updated!")
                st.rerun()
        
        # Quick Stats
        st.markdown("### 📊 Quick Stats")
        col1, col2 = st.columns(2)
        
        with col1:
            emotions_today = get_emotion_history(user_id, 1)
            emotion = emotions_today.iloc[0]['emotion'] if not emotions_today.empty else "Calm"
            st.metric("Today's Mood", emotion)
        
        with col2:
            calories_today = get_nutrition_history(user_id, 1)['calories'].sum() if not get_nutrition_history(user_id, 1).empty else 0
            st.metric("Calories", f"{calories_today}")
        
        # Navigation
        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        
        # Custom tab selection
        tabs = ["🏠 Dashboard", "🎤 Voice Check-in", "📝 Text Analysis", "👶 Baby Tracker", 
                "🍎 Nutrition", "💪 Exercise", "💊 Vitamins", "💡 Recommendations", "📄 Reports"]
        
        selected_tab = st.radio("Go to:", tabs, key="tab_navigation")
        
        # Update current tab based on selection
        tab_mapping = {
            "🏠 Dashboard": "Dashboard",
            "🎤 Voice Check-in": "Voice",
            "📝 Text Analysis": "Text",
            "👶 Baby Tracker": "Baby",
            "🍎 Nutrition": "Nutrition",
            "💪 Exercise": "Exercise",
            "💊 Vitamins": "Vitamins",
            "💡 Recommendations": "Recommendations",
            "📄 Reports": "Reports"
        }
        
        st.session_state.current_tab = tab_mapping[selected_tab]
        
        # Emergency Information
        st.markdown("---")
        with st.expander("🆘 Emergency Support"):
            st.markdown("""
            **Seek immediate medical attention for:**
            - Severe abdominal pain
            - Heavy bleeding
            - Decreased fetal movement
            - Signs of preeclampsia
            
            **General Emergency Numbers (Pakistan):**
            - Unified Emergency Helpline (Police, Ambulance, Fire): 911
            - Police Emergency: 15
            - Rescue Services (Ambulance / Fire / Rescue 1122): 1122
            - Fire Brigade: 16
            - Edhi Ambulance: 115 (may vary locally)
            - Chhipa Ambulance: 1020
            - Medical Helpline: 1166
            - National Highway/Motorway Police: 130
            """)
        
        # Logout
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.login = False
            st.session_state.user = None
            st.session_state.page = "login"
            st.rerun()    
    
    # Main Content based on selected tab
    st.markdown('<h1 class="main-header">🤰 Pregnancy Wellness Assistant</h1>', unsafe_allow_html=True)
    
    # Tab Content
    if st.session_state.current_tab == "Dashboard":
        show_dashboard_tab(user_id, username, trimester, weeks_pregnant, baby_name)
    elif st.session_state.current_tab == "Voice":
        show_voice_tab(user_id, weeks_pregnant, baby_name)
    elif st.session_state.current_tab == "Text":
        show_text_analysis_tab(user_id)
    elif st.session_state.current_tab == "Baby":
        show_baby_tracker_tab(user_id, weeks_pregnant, baby_name)
    elif st.session_state.current_tab == "Nutrition":
        show_nutrition_tab(user_id, trimester, weeks_pregnant)
    elif st.session_state.current_tab == "Exercise":
        show_exercise_tab(user_id, trimester, weeks_pregnant)
    elif st.session_state.current_tab == "Vitamins":
        show_vitamins_tab(user_id, trimester)
    elif st.session_state.current_tab == "Recommendations":
        show_recommendations_tab(user_id, trimester, weeks_pregnant)
    elif st.session_state.current_tab == "Reports":
        show_reports_tab(user_id, username, weeks_pregnant, trimester, baby_name)

# ============================================
# TAB FUNCTIONS
# ============================================

def show_dashboard_tab(user_id, username, trimester, weeks_pregnant, baby_name):
    """Enhanced Dashboard Tab"""
    st.markdown('<h2 class="sub-header">Wellness Dashboard</h2>', unsafe_allow_html=True)
    
    # Top Stats Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        emotions_today = get_emotion_history(user_id, 1)
        emotion = emotions_today.iloc[0]['emotion'] if not emotions_today.empty else "Calm"
        st.metric("Today's Emotion", emotion)
    
    with col2:
        kicks_today = get_baby_kicks_history(user_id, 1)['kicks'].sum() if not get_baby_kicks_history(user_id, 1).empty else 0
        st.metric("Baby Kicks", kicks_today)
    
    with col3:
        calories_today = get_nutrition_history(user_id, 1)['calories'].sum() if not get_nutrition_history(user_id, 1).empty else 0
        st.metric("Calories Today", f"{calories_today}")
    
    with col4:
        exercise_today = get_exercise_history(user_id, 1)['duration_minutes'].sum() if not get_exercise_history(user_id, 1).empty else 0
        st.metric("Exercise (min)", exercise_today)
    
    # Welcome Section
    st.markdown(f"""
    <div class="info-box">
    <h3>Welcome back, {username}! 👋</h3>
    <strong>🤰 Pregnancy Progress:</strong><br>
    <strong>Baby:</strong> {baby_name}<br>
    <strong>Week:</strong> {weeks_pregnant} • <strong>Trimester:</strong> {trimester}<br>
    <strong>Today:</strong> {datetime.date.today().strftime("%B %d, %Y")}
    </div>
    """, unsafe_allow_html=True)
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        # Emotion Chart
        emotions_7d = get_emotion_history(user_id, 7)
        if not emotions_7d.empty:
            fig = px.line(emotions_7d, x='date', y='confidence', color='emotion',
                         title="Emotional Trends (7 Days)",
                         markers=True)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No emotion data yet. Complete a check-in!")
    
    with col2:
        # Nutrition Chart
        nutrition_7d = get_nutrition_history(user_id, 7)
        if not nutrition_7d.empty:
            nutrition_7d['date'] = pd.to_datetime(nutrition_7d['date'])
            daily_calories = nutrition_7d.groupby('date')['calories'].sum().reset_index()
            
            fig = px.bar(daily_calories, x='date', y='calories',
                        title="Daily Calories (7 Days)",
                        color='calories',
                        color_continuous_scale='viridis')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Log your meals to see nutrition trends!")
    
    # Baby Development
    st.markdown(f"### 👶 Baby Development - Week {weeks_pregnant}")
    baby_info = get_baby_info_by_week(weeks_pregnant)
    st.markdown(f"""
    <div class="baby-box">
    {baby_info}
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    qcol1, qcol2, qcol3 = st.columns(3)
    
    with qcol1:
        if st.button("🎤 Voice Check-in", use_container_width=True):
            st.session_state.current_tab = "Voice"
            st.rerun()
    
    with qcol2:
        if st.button("📝 Log Meal", use_container_width=True):
            st.session_state.current_tab = "Nutrition"
            st.rerun()
    
    with qcol3:
        if st.button("💪 Log Exercise", use_container_width=True):
            st.session_state.current_tab = "Exercise"
            st.rerun()

def show_voice_tab(user_id, weeks_pregnant, baby_name):
    """Voice Analysis Tab - Audio Only"""
    st.markdown('<h2 class="sub-header">Voice Emotional Check-in</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <strong>How it works:</strong><br>
        1. Upload an audio file of your voice<br>
        2. The AI analyzes your emotional state from your voice<br>
        3. Get personalized recommendations<br>
        4. Track your emotional wellness over time
        </div>
        """, unsafe_allow_html=True)
        
        # Audio file upload
        st.markdown("### 🎤 Upload Your Voice")
        uploaded_file = st.file_uploader(
            "Choose an audio file (WAV, MP3, M4A)", 
            type=['wav', 'mp3', 'm4a'],
            key="audio_uploader"
        )
        
        if uploaded_file:
            # Display file info
            file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
            st.info(f"📁 File uploaded: {uploaded_file.name} ({file_size:.2f} MB)")
            
            # Preview audio
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("🧠 Analyze Voice Emotion", use_container_width=True, type="primary"):
                with st.spinner("Analyzing your voice emotion..."):
                    try:
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Predict emotion
                        emotion, confidence, scores, raw_emotion, model_type = voice_detector.predict_emotion(tmp_path)
                        
                        # Save to session state
                        st.session_state.current_emotion = {
                            'emotion': emotion,
                            'confidence': confidence,
                            'scores': scores,
                            'raw_emotion': raw_emotion,
                            'model_type': model_type,
                            'timestamp': datetime.datetime.now(),
                            'source': 'voice'
                        }
                        
                        # Save to database
                        add_emotion(user_id, emotion, confidence, "voice_analysis", "")
                        
                        # Load audio for visualization
                        try:
                            y, sr = librosa.load(tmp_path, sr=16000, duration=3)
                            st.session_state.audio_data = (y, sr)
                        except:
                            st.session_state.audio_data = None
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                        
                        st.success("Voice analysis complete!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")
                        if 'tmp_path' in locals():
                            os.unlink(tmp_path)
        
        # Tips for better recording
        with st.expander("💡 Tips for Better Voice Analysis"):
            st.markdown("""
            **For best results:**
            
            1. **Record clearly** - Speak in a normal tone
            2. **Duration** - 3-10 seconds is ideal
            3. **Content** - Describe how you're feeling
            4. **Environment** - Quiet background, minimal noise
            5. **Examples of what to say:**
               - "I'm feeling happy and excited today"
               - "I've been feeling a bit anxious lately"
               - "Today I'm tired but content"
            
            **Supported formats:** WAV, MP3, M4A
            """)
    
    with col2:
        # Display results
        if st.session_state.current_emotion and st.session_state.current_emotion.get('source') == 'voice':
            emotion = st.session_state.current_emotion['emotion']
            confidence = st.session_state.current_emotion['confidence']
            scores = st.session_state.current_emotion['scores']
            model_type = st.session_state.current_emotion['model_type']
            
            st.markdown(f"""
            <div class="success-box">
            <h3>🎯 Analysis Results</h3>
            <strong>Detected Emotion:</strong> {emotion}<br>
            <strong>Confidence:</strong> {confidence:.1%}<br>
            <strong>Method:</strong> {model_type}<br>
            <strong>Time:</strong> {st.session_state.current_emotion['timestamp'].strftime('%H:%M')}
            </div>
            """, unsafe_allow_html=True)
            
            # Radar chart
            if scores:
                st.markdown("#### 📊 Emotion Distribution")
                radar_fig = plot_emotion_radar(scores)
                st.plotly_chart(radar_fig, use_container_width=True)
            
            # Audio visualization
            if st.session_state.audio_data and st.session_state.current_emotion.get('source') == 'voice':
                y, sr = st.session_state.audio_data
                wave_fig = plot_audio_waveform(y, sr)
                st.plotly_chart(wave_fig, use_container_width=True)
            
            # Baby connection tip
            st.markdown(f"""
            <div class="baby-box">
            <strong>👶 Baby Connection Tip:</strong><br>
            Share your feeling of "{emotion.lower()}" with {baby_name} today. 
            Your voice and emotions help your baby develop emotional awareness.
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.info("👆 Upload an audio file to analyze your voice emotion.")
def show_text_analysis_tab(user_id):
    """Text-based Emotion Analysis Tab"""
    st.markdown('<h2 class="sub-header">Text Emotion Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>How it works:</strong><br>
    1. Describe your feelings, thoughts, or mood<br>
    2. Our AI model analyzes the text<br>
    3. Get insights into your emotional state<br>
    4. Receive personalized recommendations
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input
        user_text = st.text_area(
            "How are you feeling today?",
            placeholder="Describe your feelings, thoughts, or mood...\n\nExample: 'I feel anxious about my upcoming appointment but also excited to see the baby.'",
            height=200,
            key="text_analysis_input"
        )
        
        notes = st.text_input("Additional context (optional):", key="text_notes")
        
        if st.button("🧠 Analyze Text Emotion", use_container_width=True, type="primary"):
            if user_text:
                with st.spinner("Analyzing your text..."):
                    # Use the text emotion detector
                    emotion, confidence, scores, model_type = text_detector.analyze_text(user_text)
                    
                    # Save to session state
                    st.session_state.current_emotion = {
                        'emotion': emotion,
                        'confidence': confidence,
                        'scores': scores,
                        'model_type': model_type,
                        'timestamp': datetime.datetime.now(),
                        'source': 'text'
                    }
                    
                    # Save to database
                    add_emotion(user_id, emotion, confidence, "text_analysis", notes)
                    
                    st.success("Analysis complete!")
                    st.rerun()
            else:
                st.error("Please enter some text to analyze.")
    
    with col2:
        # Display results
        if st.session_state.current_emotion and st.session_state.current_emotion.get('source') == 'text':
            emotion = st.session_state.current_emotion['emotion']
            confidence = st.session_state.current_emotion['confidence']
            scores = st.session_state.current_emotion['scores']
            model_type = st.session_state.current_emotion['model_type']
            
            st.markdown(f"""
            <div class="success-box">
            <h3>📝 Analysis Results</h3>
            <strong>Detected Emotion:</strong> {emotion}<br>
            <strong>Confidence:</strong> {confidence:.1%}<br>
            <strong>Model:</strong> {model_type}<br>
            <strong>Time:</strong> {st.session_state.current_emotion['timestamp'].strftime('%H:%M')}
            </div>
            """, unsafe_allow_html=True)
            
            # Emotion distribution
            if scores:
                st.markdown("#### 📊 Emotion Distribution")
                fig = go.Figure(data=[
                    go.Bar(x=list(scores.keys()), y=list(scores.values()),
                          marker_color='#9370DB')
                ])
                fig.update_layout(height=300, xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Text insights
            st.markdown("#### 💭 Text Insights")
            insights = {
                'Happy': "Great! Positive emotions are wonderful for you and baby.",
                'Calm': "Peaceful moments are precious during pregnancy.",
                'Anxious': "It's normal to feel anxious. Try deep breathing exercises.",
                'Sad': "Pregnancy hormones can affect mood. Be gentle with yourself.",
                'Angry': "Frustration is common. Try talking about your feelings.",
                'Surprised': "Pregnancy brings many surprises! Embrace the journey.",
                'Neutral': "A balanced emotional state is healthy."
            }
            
            st.info(insights.get(emotion, "Your emotional awareness is important for your wellbeing."))
            
        else:
            st.info("👆 Enter text above to get started!")

def show_baby_tracker_tab(user_id, weeks_pregnant, baby_name):
    """Baby Tracker Tab"""
    st.markdown('<h2 class="sub-header">👶 Baby Development & Tracking</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Current week info
        st.markdown(f"""
        <div class="baby-box">
        <h3>Week {weeks_pregnant} Development</h3>
        {get_baby_info_by_week(weeks_pregnant)}
        </div>
        """, unsafe_allow_html=True)
        
        # Week explorer
        st.markdown("### 🔍 Explore Different Weeks")
        explore_week = st.slider(
            "Select week to explore:",
            min_value=1,
            max_value=40,
            value=weeks_pregnant,
            key="explore_week"
        )
        
        if explore_week != weeks_pregnant:
            st.markdown(f"""
            <div class="info-box">
            {get_baby_info_by_week(explore_week)}
            </div>
            """, unsafe_allow_html=True)
        
        # Size visualization
        st.markdown("### 📏 Baby Size Comparison")
        
        size_milestones = {
            4: ("Blueberry", "🔵", "Tiny but growing!"),
            8: ("Raspberry", "🟣", "All organs forming"),
            12: ("Lime", "🟢", "First trimester complete"),
            16: ("Avocado", "🟤", "Can hear your voice"),
            20: ("Banana", "🟡", "Midpoint of pregnancy"),
            24: ("Corn", "🌽", "Viability milestone"),
            28: ("Eggplant", "🍆", "Third trimester begins"),
            32: ("Squash", "🎃", "Getting ready for birth"),
            36: ("Watermelon", "🍉", "Full term reached"),
            40: ("Pumpkin", "🎃", "Ready to meet you!")
        }
        
        closest = min(size_milestones.keys(), key=lambda x: abs(x - weeks_pregnant))
        fruit, emoji, note = size_milestones[closest]
        
        st.markdown(f"""
        <div class="metric-card">
        <div style="text-align: center;">
        <span style="font-size: 3rem;">{emoji}</span>
        <h3>{fruit}</h3>
        <p>{note}</p>
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Baby kick tracker
        if weeks_pregnant >= 16:
            st.markdown("### 👣 Baby Kick Counter")
            
            col_a, col_b = st.columns(2)
            with col_a:
                kicks = st.number_input("Kicks:", min_value=0, max_value=100, value=10, key="kicks")
            with col_b:
                duration = st.number_input("Minutes:", min_value=1, max_value=60, value=10, key="duration")
            
            kick_notes = st.text_input("Notes (optional):", key="kick_notes")
            
            if st.button("💖 Log Kicks", use_container_width=True):
                add_baby_kick(user_id, kicks, duration, kick_notes)
                st.success(f"Logged {kicks} kicks!")
                st.rerun()
            
            # Today's summary
            today_kicks_df = get_baby_kicks_history(user_id, 1)
            
            if not today_kicks_df.empty:
                total_today = today_kicks_df['kicks'].sum()
                st.markdown(f"""
                <div class="metric-card">
                <h3>Today's Activity</h3>
                <p style="font-size: 2rem; text-align: center; color: #FF69B4;">{total_today}</p>
                <p style="text-align: center;">kicks tracked</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No kicks logged today yet.")
        else:
            st.markdown("### 👣 Early Pregnancy")
            st.info(f"""
            **Baby movements start around week 16-20.**
            
            You're at week {weeks_pregnant}. 
            Expect fluttering sensations soon!
            """)

def show_nutrition_tab(user_id, trimester, week):
    """Nutrition Tracking and Recommendations Tab"""
    st.markdown('<h2 class="sub-header">🍎 Nutrition & Diet Tracker</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Quick Log
        st.markdown("### 📝 Quick Log")
        
        meal_type = st.selectbox(
            "Meal Type:",
            ["Breakfast", "Lunch", "Dinner", "Snack", "Other"],
            key="meal_type"
        )
        
        food_items = st.text_input("Food Items:", placeholder="e.g., Oatmeal, banana, milk", key="food_items")
        
        calories = st.number_input("Calories (approx):", min_value=0, max_value=2000, value=300, key="calories")
        
        nutrients = st.multiselect(
            "Main Nutrients:",
            ["Protein", "Carbs", "Fats", "Fiber", "Iron", "Calcium", "Vitamin C", "Folate"],
            key="nutrients"
        )
        
        notes = st.text_input("Notes (optional):", key="nutrition_notes")
        
        if st.button("➕ Log Meal", use_container_width=True):
            add_nutrition_log(user_id, meal_type, food_items, calories, ", ".join(nutrients), notes)
            st.success(f"Logged {meal_type}!")
            st.rerun()
        
        # Today's summary
        st.markdown("---")
        st.markdown("### 📊 Today's Nutrition")
        
        today_nutrition = get_nutrition_history(user_id, 1)
        if not today_nutrition.empty:
            total_calories = today_nutrition['calories'].sum()
            meal_count = len(today_nutrition)
            
            st.metric("Total Calories", total_calories)
            st.metric("Meals Logged", meal_count)
        else:
            st.info("No meals logged today yet.")
    
    with col2:
        # Recommendations
        st.markdown("### 💡 Personalized Recommendations")
        
        nutrition_info, week_tips = get_nutrition_recommendations(trimester, week)
        
        st.markdown(f"""
        <div class="nutrition-box">
        <h4>Essential Nutrients (Trimester {trimester})</h4>
        """, unsafe_allow_html=True)
        
        for item in nutrition_info["essential"]:
            st.write(f"• {item}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Food Recommendations
        st.markdown("#### 🥗 Recommended Foods")
        cols = st.columns(2)
        for i, food in enumerate(nutrition_info["foods"]):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="background: #f0f8ff; padding: 10px; border-radius: 10px; margin: 5px 0;">
                {food}
                </div>
                """, unsafe_allow_html=True)
        
        # Tips
        st.markdown("#### 💡 Nutrition Tips")
        for tip in nutrition_info["tips"]:
            st.write(f"• {tip}")
        
        if week_tips:
            st.markdown("#### 📅 Week-Specific Tips")
            for tip in week_tips:
                st.info(f"• {tip}")

def show_exercise_tab(user_id, trimester, week):
    """Exercise Tracking and Recommendations Tab"""
    st.markdown('<h2 class="sub-header">💪 Exercise & Activity Tracker</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Quick Log
        st.markdown("### 📝 Quick Log")
        
        exercise_type = st.selectbox(
            "Exercise Type:",
            ["Walking", "Prenatal Yoga", "Swimming", "Light Strength", "Stretching", 
             "Stationary Bike", "Pilates", "Dancing", "Other"],
            key="exercise_type"
        )
        
        duration = st.number_input("Duration (minutes):", min_value=1, max_value=180, value=30, key="duration")
        
        intensity = st.select_slider(
            "Intensity:",
            options=["Very Light", "Light", "Moderate", "Vigorous"],
            value="Moderate",
            key="intensity"
        )
        
        calories = st.number_input("Calories Burned (approx):", min_value=0, max_value=1000, value=150, key="exercise_calories")
        
        notes = st.text_input("Notes (optional):", key="exercise_notes")
        
        if st.button("➕ Log Exercise", use_container_width=True):
            add_exercise_log(user_id, exercise_type, duration, intensity, calories, notes)
            st.success(f"Logged {exercise_type}!")
            st.rerun()
        
        # Today's summary
        st.markdown("---")
        st.markdown("### 📊 Today's Activity")
        
        today_exercise = get_exercise_history(user_id, 1)
        if not today_exercise.empty:
            total_minutes = today_exercise['duration_minutes'].sum()
            total_calories = today_exercise['calories_burned'].sum()
            
            st.metric("Total Minutes", total_minutes)
            st.metric("Calories Burned", total_calories)
        else:
            st.info("No exercise logged today yet.")
    
    with col2:
        # Recommendations
        st.markdown("### 💡 Personalized Recommendations")
        
        exercise_info, week_tips = get_exercise_recommendations(trimester, week)
        
        st.markdown(f"""
        <div class="exercise-box">
        <h4>Safe Exercises (Trimester {trimester})</h4>
        """, unsafe_allow_html=True)
        
        for item in exercise_info["safe"]:
            st.write(f"• {item}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Benefits
        st.markdown("#### 🌟 Exercise Benefits")
        cols = st.columns(2)
        for i, benefit in enumerate(exercise_info["benefits"]):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="background: #e8f4fd; padding: 10px; border-radius: 10px; margin: 5px 0;">
                {benefit}
                </div>
                """, unsafe_allow_html=True)
        
        # Avoid
        st.markdown("#### ⚠️ Exercises to Avoid")
        for item in exercise_info["avoid"]:
            st.write(f"• {item}")
        
        if week_tips:
            st.markdown("#### 📅 Week-Specific Tips")
            for tip in week_tips:
                st.info(f"• {tip}")

def show_vitamins_tab(user_id, trimester):
    """Vitamin and Supplement Tracking Tab"""
    st.markdown('<h2 class="sub-header">💊 Vitamins & Supplements</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Quick Log
        st.markdown("### 📝 Quick Log")
        
        vitamin_name = st.selectbox(
            "Vitamin/Supplement:",
            ["Prenatal Multivitamin", "Folic Acid", "Iron", "Calcium", "Vitamin D", 
             "Omega-3 DHA", "Vitamin C", "Magnesium", "Probiotics", "Other"],
            key="vitamin_name"
        )
        
        taken = st.radio("Taken today?", ["Yes", "No"], horizontal=True, key="vitamin_taken")
        
        dosage = st.text_input("Dosage (optional):", placeholder="e.g., 400 mcg", key="vitamin_dosage")
        
        notes = st.text_input("Notes (optional):", key="vitamin_notes")
        
        if st.button("➕ Log Vitamin", use_container_width=True):
            add_vitamin_log(user_id, vitamin_name, taken == "Yes", dosage, notes)
            st.success(f"Logged {vitamin_name}!")
            st.rerun()
        
        # Today's summary
        st.markdown("---")
        st.markdown("### 📊 Today's Intake")
        
        today_vitamins = get_vitamin_history(user_id, 1)
        if not today_vitamins.empty:
            taken_count = today_vitamins['taken'].sum()
            total_count = len(today_vitamins)
            
            st.metric("Taken Today", f"{taken_count}/{total_count}")
        else:
            st.info("No vitamins logged today yet.")
    
    with col2:
        # Recommendations
        st.markdown("### 💡 Vitamin Recommendations")
        
        vitamins_info, extra_tips = get_vitamin_recommendations(trimester)
        
        st.markdown("#### 💊 Essential Vitamins")
        for item in vitamins_info["essential"]:
            st.write(f"• {item}")
        
        st.markdown("#### 🌿 Optional Supplements")
        for item in vitamins_info["optional"]:
            st.write(f"• {item}")
        
        st.markdown("#### 💡 Important Tips")
        for tip in vitamins_info["tips"]:
            st.info(f"• {tip}")
        
        if extra_tips:
            st.markdown("#### 📅 Trimester-Specific")
            for tip in extra_tips:
                st.success(f"• {tip}")

def show_recommendations_tab(user_id, trimester, week):
    """Enhanced Recommendations Tab with all categories"""
    st.markdown('<h2 class="sub-header">💡 Personalized Recommendations</h2>', unsafe_allow_html=True)
    
    # Get latest emotion
    emotions_df = get_emotion_history(user_id, 1)
    if not emotions_df.empty:
        latest_emotion = emotions_df.iloc[0]['emotion']
        latest_confidence = emotions_df.iloc[0]['confidence']
    else:
        latest_emotion = "Calm"
        latest_confidence = 0.5
    
    # Display summary
    st.markdown(f"""
    <div class="info-box">
    <h3>Personalized for You</h3>
    <strong>Current Emotion:</strong> {latest_emotion} ({latest_confidence:.0%} confidence)<br>
    <strong>Pregnancy Week:</strong> {week}<br>
    <strong>Trimester:</strong> {trimester}
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different recommendation types
    rec_tab1, rec_tab2, rec_tab3, rec_tab4 = st.tabs(["🎭 Emotional", "🍎 Nutritional", "💪 Exercise", "💊 Vitamins"])
    
    with rec_tab1:
        st.markdown("#### 🎭 Emotional Wellness")
        emotional_recs = get_emotional_recommendations(latest_emotion, trimester, week)
        for i, rec in enumerate(emotional_recs, 1):
            st.markdown(f"**{i}.** {rec}")
    
    with rec_tab2:
        st.markdown("#### 🍎 Nutritional Guidance")
        nutrition_info, _ = get_nutrition_recommendations(trimester, week)
        st.markdown("**Essential Nutrients:**")
        for item in nutrition_info["essential"][:3]:
            st.write(f"• {item}")
        
        st.markdown("**Recommended Foods:**")
        for item in nutrition_info["foods"][:3]:
            st.write(f"• {item}")
    
    with rec_tab3:
        st.markdown("#### 💪 Exercise Suggestions")
        exercise_info, _ = get_exercise_recommendations(trimester, week)
        st.markdown("**Safe Exercises:**")
        for item in exercise_info["safe"][:3]:
            st.write(f"• {item}")
        
        st.markdown("**Benefits:**")
        for item in exercise_info["benefits"][:2]:
            st.write(f"• {item}")
    
    with rec_tab4:
        st.markdown("#### 💊 Vitamin Recommendations")
        vitamins_info, _ = get_vitamin_recommendations(trimester)
        st.markdown("**Essential Vitamins:**")
        for item in vitamins_info["essential"]:
            st.write(f"• {item}")

def show_reports_tab(user_id, username, weeks_pregnant, trimester, baby_name):
    """Enhanced Reports Tab"""
    st.markdown('<h2 class="sub-header">📄 Reports & Data Export</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Wellness Report
        st.markdown("### 📊 Generate Wellness Report")
        
        report_type = st.selectbox(
            "Report Type:",
            ["Weekly Summary", "Monthly Overview", "Trimester Progress", "Complete History"],
            key="report_type"
        )
        
        if st.button("📋 Generate PDF Report", use_container_width=True):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            # Header
            pdf.cell(200, 10, txt="Pregnancy Wellness Report", ln=1, align='C')
            pdf.cell(200, 10, txt=f"For: {username}", ln=1)
            pdf.cell(200, 10, txt=f"Date: {datetime.date.today()}", ln=1)
            pdf.cell(200, 10, txt=f"Baby: {baby_name} - Week {weeks_pregnant}, Trimester {trimester}", ln=1)
            pdf.ln(10)
            
            # Add sections
            if report_type != "Complete History":
                days = 7 if report_type == "Weekly Summary" else 30 if report_type == "Monthly Overview" else 90
                
                # Emotions
                pdf.cell(200, 10, txt="Recent Emotions:", ln=1)
                emotions = get_emotion_history(user_id, days)
                for _, row in emotions.iterrows():
                    pdf.cell(200, 10, txt=f"{row['date']}: {row['emotion']} ({row['confidence']:.0%})", ln=1)
                
                pdf.ln(5)
                
                # Nutrition
                pdf.cell(200, 10, txt="Nutrition Summary:", ln=1)
                nutrition = get_nutrition_history(user_id, days)
                if not nutrition.empty:
                    total_calories = nutrition['calories'].sum()
                    pdf.cell(200, 10, txt=f"Total Calories: {total_calories}", ln=1)
                
                pdf.ln(5)
                
                # Exercise
                pdf.cell(200, 10, txt="Exercise Summary:", ln=1)
                exercise = get_exercise_history(user_id, days)
                if not exercise.empty:
                    total_minutes = exercise['duration_minutes'].sum()
                    pdf.cell(200, 10, txt=f"Total Exercise: {total_minutes} minutes", ln=1)
            
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=f"wellness_report_{datetime.date.today()}.pdf",
                mime="application/pdf"
            )
    
    with col2:
        # Data Export
        st.markdown("### 💾 Export Data")
        
        export_format = st.selectbox(
            "Format:",
            ["JSON", "CSV"],
            key="export_format"
        )
        
        if st.button("📤 Export All Data", use_container_width=True):
            export_data = {
                'user_info': st.session_state.user,
                'emotions': get_emotion_history(user_id, 365).to_dict('records'),
                'baby_kicks': get_baby_kicks_history(user_id, 365).to_dict('records'),
                'nutrition': get_nutrition_history(user_id, 365).to_dict('records'),
                'exercise': get_exercise_history(user_id, 365).to_dict('records'),
                'vitamins': get_vitamin_history(user_id, 365).to_dict('records'),
                'export_date': datetime.datetime.now().isoformat()
            }
            
            if export_format == "JSON":
                export_json = json.dumps(export_data, indent=2, default=str)
                
                st.download_button(
                    label="⬇️ Download JSON",
                    data=export_json,
                    file_name=f"pregnancy_data_{datetime.date.today()}.json",
                    mime="application/json"
                )
    
    # Statistics
    st.markdown("---")
    st.markdown("### 📊 Your Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_emotions = len(get_emotion_history(user_id, 365))
        st.metric("Total Emotions", total_emotions)
    
    with col2:
        total_kicks = get_baby_kicks_history(user_id, 365)['kicks'].sum() if not get_baby_kicks_history(user_id, 365).empty else 0
        st.metric("Total Kicks", total_kicks)
    
    with col3:
        total_meals = len(get_nutrition_history(user_id, 365))
        st.metric("Meals Logged", total_meals)
    
    with col4:
        total_exercise = get_exercise_history(user_id, 365)['duration_minutes'].sum() if not get_exercise_history(user_id, 365).empty else 0
        st.metric("Exercise Minutes", total_exercise)

# ============================================
# MAIN APP ROUTING
# ============================================
def main():
    if st.session_state.page == "login":
        show_login_page()
    elif st.session_state.page == "signup":
        show_signup_page()
    elif st.session_state.page == "main":
        show_main_app()
    else:
        st.session_state.page = "login"
        show_login_page()

# ============================================
# RUN THE APP
# ============================================
if __name__ == "__main__":
    main()
    
    # Footer
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.caption("🤰 Pregnancy Wellness Assistant")
        st.caption("Version 1.0.0")
    
    with footer_col2:
        st.caption("💖 Supporting maternal mental health")
        st.caption("Not a substitute for medical care")
    
    with footer_col3:
        st.caption("🔒 Your data is stored locally")
        st.caption("Built with Streamlit & SQLite")

# Close database connection
conn.close()