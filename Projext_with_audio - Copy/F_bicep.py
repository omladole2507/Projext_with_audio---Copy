import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading  # Added for multithreading

# Initialize Mediapipe Pose Tools
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def speak(message):
    """Function to speak the given message in a separate thread."""
    def _speak():
        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()
    
    # Start a new thread for audio feedback
    threading.Thread(target=_speak).start()

# Rest of the code remains the same...

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    """
    Calculate the angle between three points
    using the law of cosines.
    :param a: Shoulder point
    :param b: Elbow joint
    :param c: Wrist point
    :return: Angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Calculate the distance vectors
    ab = a - b
    cb = c - b

    # Use dot product and arccosine to calculate the angle
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)  # Avoid division by zero
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    return angle

# Main function for real-time bicep curl detection
def count_curls():
    """
    Function to detect arm curls using webcam feed.
    It calculates bicep angles and counts repetitions based on motion.
    """
    cap = cv2.VideoCapture(0)  # Open webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    counter = 0  # Repetition counter
    stage = None  # Track motion stage
    alert_message = ""

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for mirror-like view
            frame = cv2.flip(frame, 1)

            # Convert to RGB and process with Mediapipe Pose
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # Extract landmarks for analysis
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Define shoulder, elbow, wrist points for angle calculations
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

                # Calculate angles
                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                # Determine good/poor posture feedback
                if abs(left_angle - right_angle) > 20:
                    alert_message = "Incorrect form! Adjust your arms!"
                    speak(alert_message)  # Speak the alert message
                else:
                    alert_message = "Good posture!"

                # Determine curl stage
                if right_angle > 160:
                    stage = "down"
                if right_angle < 30 and stage == "down":
                    stage = "up"
                    counter += 1
                    alert_message = f"Great! Reps: {counter}"
                    speak(alert_message)  # Speak the rep count

                # Display angles and feedback
                cv2.putText(frame, f"Left Arm Angle: {int(left_angle)}", (15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                cv2.putText(frame, f"Right Arm Angle: {int(right_angle)}", (15, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                cv2.putText(frame, f"Reps: {counter}", (15, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                cv2.putText(frame, alert_message, (15, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                # Draw Mediapipe landmarks on feed
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                )

            # Stream video to the user
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()