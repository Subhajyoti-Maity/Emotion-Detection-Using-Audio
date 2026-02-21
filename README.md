
# 🚀 Installation

Clone the repository from GitHub:


# 🎭 Emotion Detection Using Audio 🎙️

Detect emotions from your voice in real time or from uploaded audio files using a deep learning model. This project features a modern web interface powered by Flask and a pre-trained neural network.

---

## 🚀 Features

- 🎤 Real-time audio recording and analysis
- 📁 Upload audio files for emotion detection
- 🤖 Deep learning model (Keras/TensorFlow, .h5 format)
- 🧠 MFCC feature extraction (Librosa)
- 📊 Detects 8 emotions: Angry, Happy, Neutral, Sad, Calm, Fearful, Disgust, Surprised
- 💻 Clean, responsive UI (HTML/CSS/JS)




---

## 📈 Model Accuracy

**Overall model accuracy:** 78–85% (on standard datasets; real-world results may vary)

---


## 📂 Project Structure


```
Emotion-Detection-Using-Audio/
│
├── app.py                # Flask backend and logic
├── emotion_model.h5      # Pre-trained Keras model
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore file
├── .venv/                # Python virtual environment (recommended)
├── .vscode/              # VS Code settings (optional)
├── static/               # Uploaded audio files for emotion detection
├── demo_clips/           # Application demo clips

├── templates/
│   └── index.html        # Main web UI
```

---

## 🎬 Demo Clips

Below are demo screenshots of the application in action:

[Emotion detection uploding audio.png](demo_clips/Emotion%20detection%20uploding%20audio.png)

[Real time audio recording emotion detection.png](demo_clips/Real%20time%20audio%20recording%20emotion%20detection.png)

---

## ⚡ Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Subhajyoti-Maity/Emotion-Detection-Using-Audio.git
cd Emotion-Detection-Using-Audio
```

### 2. Create and activate a virtual environment (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
.venv\Scripts\python.exe app.py
# or, after activating venv:
set FLASK_APP=app.py
flask run
```

Visit [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

---

## 📝 Usage

1. Open the web app in your browser.
2. Click "Record" to analyze your voice in real time, or upload an audio file.
3. The detected emotion and confidence will be displayed instantly.
4. All 8 emotions are supported and shown in the UI.

---

## 🛠️ Requirements

- Python 3.8+
- Flask
- TensorFlow / Keras
- Librosa
- Numpy

Install all dependencies with `pip install -r requirements.txt`.

---

## 📢 Contributing

Pull requests and suggestions are welcome! Please open an issue for major changes.

---



## 📞 Support & Contact

- **GitHub Issues:** Report bugs or request features
- **Email:** Please contact via your GitHub profile or open an issue for support.
- **Documentation:** Refer to code comments and API route docs in the repository.

## 👤 Author

Subhajyoti Maity
