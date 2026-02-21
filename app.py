from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import numpy as np
try:
    from keras.models import load_model
except ImportError:
    from keras.models import load_model
import librosa
from pydub import AudioSegment
import random
import io
import base64
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Emotion labels (update if your model uses different order)
emotion_labels = ['angry', 'happy', 'neutral', 'sad', 'calm', 'fearful', 'disgust', 'surprised']

# Load the trained model once at startup
MODEL_PATH = 'emotion_model.h5'
print(f"Loading model from: {MODEL_PATH}")
try:
    emotion_model = load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    emotion_model = None

def preprocess_audio(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    # Supported formats: wav, mp3, webm, mp4, m4a, ogg, flac
    if ext != '.wav':
        audio = AudioSegment.from_file(filepath)
        wav_path = filepath.replace(ext, '.wav')
        audio.export(wav_path, format='wav')
        filepath = wav_path
    y, sr = librosa.load(filepath, sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Restore to 40 for model compatibility
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return np.expand_dims(mfcc_scaled, axis=0)

def preprocess_audio_from_bytes(audio_bytes, audio_format='wav'):
    """Preprocess audio from bytes (for real-time analysis)"""
    try:
        print(f"Preprocessing audio: {len(audio_bytes)} bytes, format: {audio_format}")
        
        # Use in-memory conversion for all formats except wav
        try:
            if audio_format != 'wav':
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
                buf = io.BytesIO()
                audio.export(buf, format='wav')
                buf.seek(0)
                y, sr = librosa.load(buf, sr=22050, mono=True)
            else:
                y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)
            print(f"Audio loaded: {len(y)} samples, {sr} Hz")
            if len(y) < sr * 1.0:
                print(f"Audio too short: {len(y)/sr:.2f} seconds")
                return None
            print(f"Audio length: {len(y)/sr:.2f} seconds")
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfcc_scaled = np.mean(mfcc.T, axis=0)
            features = np.expand_dims(mfcc_scaled, axis=0)
            print(f"Features extracted: shape {features.shape}")
            return features
        except Exception as e:
            import traceback
            print(f"Error preprocessing audio: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return None
                    
    except Exception as e:
        import traceback
        print(f"Error preprocessing audio: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    prediction = None
    filename = None
    import traceback
    if request.method == 'POST':
        audio = request.files['audio']
        if audio and audio.filename:
            ext = os.path.splitext(audio.filename)[1].lower()
            filepath = os.path.join('static', audio.filename)
            audio.save(filepath)
            try:
                features = preprocess_audio(filepath)
                if features is None:
                    error = 'Audio preprocessing failed. File may be too short or invalid.'
                    print(error)
                    return render_template('index.html', prediction=None, filename=audio.filename, error=error)
                prediction = emotion_model.predict(features)
                predicted_label = emotion_labels[np.argmax(prediction)]
                return render_template('index.html', prediction=predicted_label, filename=audio.filename, error=None)
            except Exception as e:
                error = f'Error processing audio: {str(e)}\n{traceback.format_exc()}'
                print(error)
                return render_template('index.html', prediction=None, filename=audio.filename, error=error)
        else:
            error = 'No audio file uploaded.'
            print(error)
            return render_template('index.html', prediction=None, filename=None, error=error)
    return render_template('index.html', prediction=None, filename=None, error=None)

@app.route('/analyze_realtime', methods=['POST'])
def analyze_realtime():
    """Handle real-time audio analysis"""
    try:
        print("=== Starting real-time analysis ===")
        
        if 'audio' not in request.files:
            print("No audio file in request")
            return jsonify({'success': False, 'error': 'No audio file provided'})
        
        audio_file = request.files['audio']
        
        # Get file extension to determine format
        filename = audio_file.filename or 'audio.wav'
        audio_format = filename.split('.')[-1].lower() if '.' in filename else 'wav'
        
        print(f"Audio file: {filename}, format: {audio_format}")
        
        # Read audio bytes
        audio_bytes = audio_file.read()
        
        if len(audio_bytes) == 0:
            print("Empty audio file received")
            return jsonify({'success': False, 'error': 'Empty audio file'})
        
        print(f"Received audio: {len(audio_bytes)} bytes, format: {audio_format}")
        
        # Preprocess audio
        print("Starting audio preprocessing...")
        features = preprocess_audio_from_bytes(audio_bytes, audio_format)
        
        if features is None:
            print("Audio preprocessing failed - returned None")
            return jsonify({'success': False, 'error': 'Failed to process audio - audio too short or invalid format'})
        
        print(f"Audio preprocessing successful, features shape: {features.shape}")
        
        # Make prediction
        print("Making prediction...")
        if emotion_model is None:
            print("Model not loaded!")
            return jsonify({'success': False, 'error': 'Model not loaded'})
        
        prediction = emotion_model.predict(features, verbose=0)  # Suppress verbose output
        predicted_label = emotion_labels[np.argmax(prediction)]
        confidence = float(np.max(prediction) * 100)
        
        print(f"Prediction: {predicted_label}, Confidence: {confidence}%")
        
        return jsonify({
            'success': True,
            'emotion': predicted_label,
            'confidence': round(confidence, 1)
        })
        
    except Exception as e:
        import traceback
        print(f"Error in analyze_realtime: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/test', methods=['GET'])
def test():
    """Simple test endpoint"""
    return jsonify({'status': 'ok', 'message': 'Server is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
