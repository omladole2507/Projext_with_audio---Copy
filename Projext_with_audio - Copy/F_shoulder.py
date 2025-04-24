import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response
import pyttsx3
import threading  # Added for multithreading

# Initialize Mediapipe Pose Tools
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

app = Flask(__name__)

# Global variables
reps_right = 0
reps_left = 0
up_right = False
up_left = False
threshold_angle = 90  # Threshold angle for incorrect form detection

# Global variables for smoothing
angle_smoothing_window = 5
right_arm_angle_history = []
left_arm_angle_history = []

def speak(message):
    """Function to speak the given message in a separate thread."""
    def _speak():
        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()
    
    # Start a new thread for audio feedback
    threading.Thread(target=_speak).start()

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def smooth_angle(angle, angle_history):
    """Smooth the angle using a moving average."""
    angle_history.append(angle)
    if len(angle_history) > angle_smoothing_window:
        angle_history.pop(0)
    return np.mean(angle_history)

def shoulder_training_logic():
    """Generate the video feed and yield frames for Flask streaming."""
    global reps_right, reps_left, up_right, up_left
    global right_arm_angle_history, left_arm_angle_history

    cap = cv2.VideoCapture(0)  # Open webcam
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break

            img = cv2.resize(img, (1280, 720))

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                landmarks = results.pose_landmarks.landmark
                h, w, _ = img.shape

                # Extract relevant landmarks
                points = {
                    11: (int(landmarks[11].x * w), int(landmarks[11].y * h)),
                    12: (int(landmarks[12].x * w), int(landmarks[12].y * h)),
                    13: (int(landmarks[13].x * w), int(landmarks[13].y * h)),
                    14: (int(landmarks[14].x * w), int(landmarks[14].y * h)),
                    23: (int(landmarks[23].x * w), int(landmarks[23].y * h)),
                    24: (int(landmarks[24].x * w), int(landmarks[24].y * h)),
                }

                # Right arm angles
                angle_1_right = calculate_angle(points[14], points[12], points[24])
                smoothed_angle_1_right = smooth_angle(angle_1_right, right_arm_angle_history)
                if not up_right and points[14][1] + 40 < points[12][1]:
                    up_right = True
                elif points[14][1] > points[12][1]:
                    if up_right:
                        reps_right += 1
                        up_right = False
                        speak(f"Great! Right arm reps: {reps_right}")  # Speak the rep count

                # Left arm angles
                angle_2_left = calculate_angle(points[13], points[11], points[23])
                smoothed_angle_2_left = smooth_angle(angle_2_left, left_arm_angle_history)
                if not up_left and points[13][1] + 40 < points[11][1]:
                    up_left = True
                elif points[13][1] > points[11][1]:
                    if up_left:
                        reps_left += 1
                        up_left = False
                        speak(f"Great! Left arm reps: {reps_left}")  # Speak the rep count

                # Display text on frame
                cv2.putText(img, f"Reps Right: {reps_right}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(img, f"Reps Left: {reps_left}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(img, f"Angle 1 Right: {smoothed_angle_1_right:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(img, f"Angle 2 Left: {smoothed_angle_2_left:.2f}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                # Check for incorrect form
                if smoothed_angle_1_right > threshold_angle:
                    cv2.putText(img, "Incorrect form: Right arm angle too obtuse", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    speak("Incorrect form: Right arm angle too obtuse")  # Speak the alert
                if smoothed_angle_2_left > threshold_angle:
                    cv2.putText(img, "Incorrect form: Left arm angle too obtuse", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    speak("Incorrect form: Left arm angle too obtuse")  # Speak the alert

            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(shoulder_training_logic(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)