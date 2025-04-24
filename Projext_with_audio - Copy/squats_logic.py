import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading  # Added for multithreading

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

def angle_btn_3points(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    radians = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def is_posture_wrong(CL, CR, DL, DR):
    reasons = []
    if CL < 170 and CR < 170 and DL < 170 and DR < 170:
        if CL < 150 and CL > 80 and DL < 150 and DL > 80 and CR < 150 and CR > 80 and DR < 150 and DR > 80:
            reasons.append("Correct Posture.")
        if CL > 150:
            reasons.append("Left Hip angle is too high.")
        if CR > 150:
            reasons.append("Right Hip angle is too high.")
        if DL > 150:
            reasons.append("Left Knee angle is too high.")
        if DR > 150:
            reasons.append("Right Knee angle is too high.")
        if CL < 80:
            reasons.append("Left Hip angle is too low.")
        if CR < 80:
            reasons.append("Right Hip angle is too low.")
    else:
        reasons.append("You are in a standing position.")

    if len(reasons) > 0:
        return True, reasons
    else:
        return False, []

def squats_training_logic():
    cap = cv2.VideoCapture(0)

    # Set the frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame horizontally for a mirrored view
            frame = cv2.flip(frame, 1)

            # Detect pose landmarks
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                kp = mp_pose.PoseLandmark

                # Get coordinates and angles
                p1 = [landmarks[kp.LEFT_SHOULDER.value].x, landmarks[kp.LEFT_SHOULDER.value].y]
                p2 = [landmarks[kp.LEFT_HIP.value].x, landmarks[kp.LEFT_HIP.value].y]
                p3 = [landmarks[kp.LEFT_KNEE.value].x, landmarks[kp.LEFT_KNEE.value].y]
                CL = angle_btn_3points(p1, p2, p3)

                p1 = [landmarks[kp.RIGHT_SHOULDER.value].x, landmarks[kp.RIGHT_SHOULDER.value].y]
                p2 = [landmarks[kp.RIGHT_HIP.value].x, landmarks[kp.RIGHT_HIP.value].y]
                p3 = [landmarks[kp.RIGHT_KNEE.value].x, landmarks[kp.RIGHT_KNEE.value].y]
                CR = angle_btn_3points(p1, p2, p3)

                p1 = [landmarks[kp.LEFT_HIP.value].x, landmarks[kp.LEFT_HIP.value].y]
                p2 = [landmarks[kp.LEFT_KNEE.value].x, landmarks[kp.LEFT_KNEE.value].y]
                p3 = [landmarks[kp.LEFT_ANKLE.value].x, landmarks[kp.LEFT_ANKLE.value].y]
                DL = angle_btn_3points(p1, p2, p3)

                p1 = [landmarks[kp.RIGHT_HIP.value].x, landmarks[kp.RIGHT_HIP.value].y]
                p2 = [landmarks[kp.RIGHT_KNEE.value].x, landmarks[kp.RIGHT_KNEE.value].y]
                p3 = [landmarks[kp.RIGHT_ANKLE.value].x, landmarks[kp.RIGHT_ANKLE.value].y]
                DR = angle_btn_3points(p1, p2, p3)

                # Render detections
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
                )

                # Adjust positions for displaying angles
                cv2.putText(frame, f'Left Hip Angle: {int(CL)}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(frame, f'Right Hip Angle: {int(CR)}', (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(frame, f'Left Knee Angle: {int(DL)}', (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(frame, f'Right Knee Angle: {int(DR)}', (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                # Check for wrong posture
                posture_wrong, reasons = is_posture_wrong(CL, CR, DL, DR)
                if posture_wrong:
                    reason_text = " posture: " + ", ".join(reasons)
                    cv2.putText(frame, reason_text, (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    speak(reason_text)  # Speak the posture feedback

            # Encode and yield the frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()