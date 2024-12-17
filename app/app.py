import os
import subprocess
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import json
from shutil import copyfile
from moviepy.editor import VideoFileClip

# Dynamically determine the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WAV_OUTPUT_FOLDER = os.path.join(BASE_DIR, 'eng/wav/1_fake')
HEARTRATE_OUTPUT_FILE = os.path.join(BASE_DIR, 'results/heartrate_scores.txt')
LIPMOTION_OUTPUT_FILE = os.path.join(BASE_DIR, 'results/lipmotion_scores.txt')

app = Flask("__main__", template_folder=os.path.join(BASE_DIR, "templates"))

# Initial script configurations
CONFIG_AUDIO = {
    "CONVERT_SCRIPT_PATH": os.path.join(BASE_DIR, 'extractAudio.py'),
    "PREPROCESS_SCRIPT_PATH": os.path.join(BASE_DIR, 'preprocess-skipper.py'),
    "VALIDATE_SCRIPT_PATH": os.path.join(BASE_DIR, 'validate_trial.py'),
    "UPLOAD_FOLDER": os.path.join(BASE_DIR, 'eng/1_fake')
}

CONFIG_NO_AUDIO = {
    "CONVERT_SCRIPT_PATH": os.path.join(BASE_DIR, 'heart.py'),
    "PREPROCESS_SCRIPT_PATH": os.path.join(BASE_DIR, 'deepframe_prep.py'),
    "VALIDATE_SCRIPT_PATH": os.path.join(BASE_DIR, 'DeepPhys.py'),
    "UPLOAD_FOLDER": os.path.join(BASE_DIR, 'eng_heart')
}

# Function to check if a video has audio
def video_has_audio(video_path):
    try:
        clip = VideoFileClip(video_path)
        return clip.audio is not None  # True if the video has audio
    except Exception as e:
        print(f"Error checking audio: {e}")
        return False

# Functions to run scripts and read prediction output
def run_convert_script(script_path):
    subprocess.run(['python', script_path], check=True)

def run_preprocessing_script(script_path):
    subprocess.run(['python', script_path], check=True)

def run_validation_script(script_path, output_file):
    subprocess.run(['python', script_path], check=True)
    with open(output_file, 'r') as f:
        lines = f.readlines()
        return float(lines[-1].strip())

@app.route('/', methods=['POST', 'GET'])
def homepage():
    return render_template('index.html')

@app.route('/Detect', methods=['POST', 'GET'])
def DetectPage():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        # Retrieve the uploaded file
        video = request.files.get('video')
        if not video or video.filename == '':
            return "No video uploaded or invalid file.", 400

        video_filename = secure_filename(video.filename)

        # Ensure the upload directories exist
        os.makedirs(CONFIG_AUDIO["UPLOAD_FOLDER"], exist_ok=True)
        os.makedirs(CONFIG_NO_AUDIO["UPLOAD_FOLDER"], exist_ok=True)

        # Save the video file in both upload folders
        audio_video_path = os.path.join(CONFIG_AUDIO["UPLOAD_FOLDER"], video_filename)
        no_audio_video_path = os.path.join(CONFIG_NO_AUDIO["UPLOAD_FOLDER"], video_filename)

        video.save(audio_video_path)
        copyfile(audio_video_path, no_audio_video_path)

        # Check for audio and set config
        video_has_audio_flag = video_has_audio(audio_video_path)

        try:
            # Run the scripts
            run_convert_script(CONFIG_NO_AUDIO["CONVERT_SCRIPT_PATH"])
            run_preprocessing_script(CONFIG_NO_AUDIO["PREPROCESS_SCRIPT_PATH"])

            # Get heartrate prediction
            heartrate_prediction = 100 * run_validation_script(
                CONFIG_NO_AUDIO["VALIDATE_SCRIPT_PATH"], HEARTRATE_OUTPUT_FILE
            )

            # Get lip motion prediction if audio exists
            if video_has_audio_flag:
                run_convert_script(CONFIG_AUDIO["CONVERT_SCRIPT_PATH"])
                run_preprocessing_script(CONFIG_AUDIO["PREPROCESS_SCRIPT_PATH"])
                lipmotion_prediction = 100 * run_validation_script(
                    CONFIG_AUDIO["VALIDATE_SCRIPT_PATH"], LIPMOTION_OUTPUT_FILE
                )
            else:
                lipmotion_prediction = None

            # Compute final prediction
            if lipmotion_prediction is not None:
                final_prediction = (heartrate_prediction + lipmotion_prediction) / 2
            else:
                final_prediction = heartrate_prediction

            # Determine the output
            if final_prediction < 50:
                output = 'REAL'
                confidence = 100 - final_prediction
            else:
                output = 'FAKE'
                confidence = final_prediction

            # Format and send response
            data = {
                'output': output,
                'confidence': f"{confidence:.2f}%"
            }
            return render_template('index.html', data=json.dumps(data))

        finally:
            # Temporary cleanup logic if needed
            pass

if __name__ == "__main__":
    # Ensure folders exist
    os.makedirs(CONFIG_AUDIO["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(CONFIG_NO_AUDIO["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(WAV_OUTPUT_FOLDER, exist_ok=True)

    app.run(port=3000)
