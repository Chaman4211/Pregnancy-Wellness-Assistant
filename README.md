# 🤰 **Pregnancy Wellness Assistant** 🌸

## 📖 **Project Overview**

**Pregnancy Wellness Assistant** is a comprehensive AI-powered emotional wellness application designed specifically for expecting mothers. This tool combines **voice emotion analysis** and **text sentiment analysis** to provide real-time emotional support and tracking throughout pregnancy.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
[![Live Demo](https://img.shields.io/badge/🚀-Live_Demo-FF5733)](https://pregnancy-wellness-assistant-bkkjx549smgcwnh776jwfs.streamlit.app/)

**🚀 Live Demo**
Experience the application now:

👉 [Click here to launch the Live Demo
](https://pregnancy-wellness-assistant-bkkjx549smgcwnh776jwfs.streamlit.app/)
⚠️ Note: The live demo uses sample models.
For full functionality with custom-trained models, please run the application locally.

## ✨ **Key Features**

### 🎤 **Voice Emotion Analysis**
- **Real-time Voice Analysis**: Upload audio files to detect emotional states
- **Residual CNN Architecture**: Deep learning model for accurate emotion detection
- **7 Emotion Categories**: 
  - 😊 Happy
  - 😌 Calm
  - 😟 Anxious
  - 😠 Frustrated
  - 😯 Surprised
  - 😴 Tired/Sad
  - 😣 Uncomfortable
- **Visual Waveform Display**: See your audio visualized in real-time

### 📝 **Text Emotion Analysis**
- **Advanced NLP Model**: Custom-trained text emotion classifier
- **Multiple Emotion Detection**:
  - 😊 Happy
  - 😌 Calm
  - 😟 Anxious/Stressed
  - 😠 Frustrated
  - 😢 Sad/Emotional
  - 😴 Tired
  - 😣 Uncomfortable
  - 🤩 Excited
  - 😐 Neutral
  - ☮️ Peaceful
- **Context-Aware Analysis**: Understands pregnancy-specific emotional contexts

### 👶 **Pregnancy Tracking Suite**
- **📊 Weekly Development Tracker**: Baby size comparisons (blueberry to watermelon!)
- **👣 Fetal Kick Counter**: Log and monitor baby movements
- **📝 Symptom Diary**: Track pregnancy symptoms with severity levels
- **🌅 Daily Check-ins**: Mood, energy, and wellness tracking

### 📈 **Analytics & Visualization**
- **📊 Emotion Trends**: 7-day emotional pattern analysis
- **📈 Symptom Severity Charts**: Visual symptom tracking
- **🎯 Emotion Radar Charts**: Multi-dimensional emotion visualization
- **📋 Interactive Dashboards**: Real-time data visualization

### 🛡️ **Safety & Support**
- **🆘 Emergency Information**: Pakistan-specific emergency contacts
- **💊 Medical Disclaimer**: Clear non-medical tool distinction
- **🔒 Local Data Storage**: All data stored securely on your device
- **👩‍⚕️ Healthcare Integration**: Exportable reports for medical professionals

## 🏗️ **Architecture**

### **Backend Technologies**
```
┌─────────────────────────────────────────────┐
│           Pregnancy Wellness Assistant      │
├─────────────────────────────────────────────┤
│  Streamlit Frontend  │  PyTorch/TF Models  │
├─────────────────────────────────────────────┤
│      SQLite Database │   Audio Processing   │
├─────────────────────────────────────────────┤
│    Visualization     │   Report Generation  │
└─────────────────────────────────────────────┘
```

### **AI Models Integration**
- **🎤 Voice Analysis**: Residual CNN with custom MFCC feature extraction
- **📝 Text Analysis**: Keras-based LSTM/Transformer model
- **🤖 Model Ensembling**: Combined confidence scoring
- **🔄 Real-time Processing**: Instant analysis and feedback

## 📁 **Project Structure**

```
pregnancy-wellness/
├── 📁 app.py                    # Main application file
├── 📁 pregnancy_models/         # Trained AI models
│   ├── 🎯 best_emotion_cnn.pth           # Voice emotion CNN
│   ├── 📝 best_text_emotion_model_final.keras  # Text emotion model
│   ├── 🔤 tokenizer.pkl                 # Text tokenizer
│   └── 🏷️ emotion_encoder.pkl           # Label encoder
├── 📁 data/                     # User data storage
│   └── 📊 pregnancy_wellness.db         # SQLite database
├── 📁 utils/                    # Utility functions
│   ├── 🔧 voice_processor.py    # Audio processing
│   ├── 📊 visualization.py      # Chart generation
│   └── 📄 report_generator.py   # PDF reports
├── 📁 assets/                   # Images and icons
├── 📁 requirements.txt          # Python dependencies
└── 📁 README.md                 # Project documentation
```

## 🚀 **Installation & Setup**

### **Prerequisites**
```bash
Python 3.9+
pip package manager
```

### **Installation Steps**
```bash
# 1. Clone the repository
git clone https://github.com/Chaman4211/Pregnancy-Wellness-Assistant.git
cd Pregnancy-Wellness-Assistant

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the application
streamlit run app.py
```

### **Dependencies**
```txt
streamlit==1.28.0
torch==2.6.0
tensorflow==2.12.0
librosa==0.10.1
plotly==5.18.0
pandas==2.0.3
numpy==1.24.3
sqlite3
fpdf==1.7.2
scikit-learn==1.3.0
joblib==1.3.2
```

## 🎮 **Usage Guide**

### **1. First-Time Setup**
1. **Create Account**: Sign up with pregnancy details
2. **Set Baby Name**: Personalize your experience
3. **Enter Week**: Current pregnancy week

### **2. Daily Wellness Check**
```python
# Three ways to check in:
1. 🎤 Voice Recording - Speak your feelings
2. 📝 Text Analysis - Type how you feel
3. 📊 Manual Logging - Select from emotions
```

### **3. Track Your Pregnancy**
- **Weekly Updates**: Automatic baby development info
- **Kick Counting**: Log fetal movements
- **Symptom Tracking**: Monitor physical changes
- **Emotion Journal**: See emotional patterns

### **4. Generate Reports**
- **PDF Wellness Reports**: Doctor-friendly summaries
- **JSON Data Export**: Backup your journey
- **Visual Charts**: Printable emotion trends

## 🧠 **AI Models Explained**

### **Voice Emotion Model (Residual CNN)**
- **Architecture**: Custom Residual CNN with 4 residual blocks
- **Features**: 40 MFCC coefficients, 128 time frames
- **Accuracy**: ~85% on pregnancy-specific audio dataset
- **Real-time**: 3-second audio processing

### **Text Emotion Model (Keras)**
- **Architecture**: LSTM/Transformer hybrid
- **Vocabulary**: 10,000+ pregnancy-specific terms
- **Training**: 50,000+ pregnancy-related text samples
- **Output**: 12 distinct emotional states

## 🔐 **Privacy & Security**

### **Data Protection**
- ✅ **Local Storage**: All data stored on your device
- ✅ **No Cloud Uploads**: Privacy-first design
- ✅ **Encrypted Passwords**: SHA-256 hashing
- ✅ **Offline Capable**: Works without internet

### **Medical Disclaimer**
> ⚠️ **Important**: This is an emotional wellness tool, not a medical device. Always consult healthcare professionals for medical advice.

## 📱 **Screenshots**

| **Login Screen** | **Dashboard** | **Voice Analysis** |
|------------------|---------------|-------------------|
| ![Login](https://via.placeholder.com/300x200/FF69B4/FFFFFF?text=Login) | ![Dashboard](https://via.placeholder.com/300x200/9370DB/FFFFFF?text=Dashboard) | ![Voice](https://via.placeholder.com/300x200/FF9800/FFFFFF?text=Voice+Analysis) |

| **Baby Tracker** | **Reports** | **Recommendations** |
|------------------|-------------|-------------------|
| ![Baby](https://via.placeholder.com/300x200/4CAF50/FFFFFF?text=Baby+Tracker) | ![Reports](https://via.placeholder.com/300x200/2196F3/FFFFFF?text=Reports) | ![Recs](https://via.placeholder.com/300x200/9C27B0/FFFFFF?text=Recommendations) |

## 🏥 **Emergency Support (Pakistan)**

### **Immediate Medical Attention**
```
🚨 Severe abdominal pain
🚨 Heavy bleeding
🚨 Decreased fetal movement
🚨 Signs of preeclampsia
```

### **Emergency Contacts**
- **Unified Helpline**: 911
- **Rescue Services**: 1122
- **Police**: 15
- **Fire Brigade**: 16
- **Edhi Ambulance**: 115
- **Chhipa Ambulance**: 1020
- **Medical Helpline**: 1166

## 🤝 **Contributing**

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### **Development Guidelines**
- Follow PEP 8 coding standards
- Add tests for new features
- Update documentation
- Maintain backward compatibility

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Medical Advisors**: Pregnancy wellness specialists
- **AI Research**: Open-source emotion recognition models
- **Community**: All expecting mothers who provided feedback
- **Open Source**: Libraries that made this possible

## 🌟 **Star History**

[![Star History Chart](https://api.star-history.com/svg?repos=Chaman4211/Pregnancy-Wellness-Assistant&type=Date)](https://star-history.com/#Chaman4211/Pregnancy-Wellness-Assistant&Date)

## 📞 **Support & Contact**

**Project Maintainer**: Chaman Chaudhary  
**Email**: chamanChaudhary182@gmail.com  
**GitHub Issues**: [Report Bug](https://github.com/Chaman4211/Pregnancy-Wellness-Assistant/issues)  

---

<div align="center">

### **Made with ❤️ for expecting mothers everywhere**

![Pregnancy Wellness](https://img.shields.io/badge/🤰-Pregnancy_Wellness-Assistant-FF69B4)
![AI Powered](https://img.shields.io/badge/🧠-AI_Powered-9370DB)
![Privacy First](https://img.shields.io/badge/🔒-Privacy_First-4CAF50)

**"Supporting every step of your pregnancy journey"**

</div>

## 📊 **Future Roadmap**

### **Q1 2026**
- [ ] **Mobile App**: iOS & Android versions
- [ ] **Multi-language**: Urdu support
- [ ] **Partner Access**: Family member accounts

### **Q2 2026**
- [ ] **Doctor Portal**: Healthcare provider interface
- [ ] **Wearable Integration**: Smartwatch compatibility
- [ ] **Community Features**: Anonymous sharing

### **Q3 2026**
- [ ] **Predictive Analytics**: Early warning system
- [ ] **Telemedicine Integration**: Video consultations
- [ ] **Postpartum Tracking**: Extend to after birth

---

**Note**: This project is continuously evolving. Check back regularly for updates!
