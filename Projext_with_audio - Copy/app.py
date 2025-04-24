from flask import Flask, render_template, Response
from F_bicep import count_curls
from F_shoulder import shoulder_training_logic
from squats_logic import squats_training_logic

app = Flask(__name__)

# Home Page
@app.route('/')
def index():
    # Home page with exercise options
    return render_template('index2.html')

# Biceps Training
@app.route('/train_biceps', methods=['POST'])
def train_biceps():
    # Render webcam feed for biceps training
    return render_template('webcam_feed.html', webcam_url="/stream_biceps")

@app.route('/stream_biceps')
def stream_biceps():
    # Stream webcam feed for biceps
    return Response(count_curls(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Shoulders Training
@app.route('/train_shoulders', methods=['POST'])
def train_shoulders():
    # Render webcam feed for shoulders training
    return render_template('webcam_feed.html', webcam_url="/stream_shoulders")

@app.route('/stream_shoulders')
def stream_shoulders():
    # Stream MJPEG video feed for shoulders directly
    return Response(shoulder_training_logic(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Squats Training
@app.route('/train_squats', methods=['POST'])
def train_squats():
    # Render webcam feed for squats training
    return render_template('webcam_feed.html', webcam_url="/stream_squats")

@app.route('/stream_squats')
def stream_squats():
    # Stream webcam feed for squats
    return Response(squats_training_logic(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Tutorials
@app.route('/tutorial_biceps')
def tutorial_biceps():
    return render_template('video_page.html', video_url="/static/videos/biceps.mp4")

@app.route('/tutorial_shoulders')
def tutorial_shoulders():
    return render_template('video_page.html', video_url="/static/videos/shoulders.mp4")

@app.route('/tutorial_squats')
def tutorial_squats():
    return render_template('video_page.html', video_url="/static/videos/squats.mp4")

if __name__ == '__main__':
    app.run(debug=True)
