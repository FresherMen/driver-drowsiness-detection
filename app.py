import os
import time
import cv2
import numpy as np

def compute_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(eye_points):
    """
    Computes the Eye Aspect Ratio (EAR) given 6 facial landmark points.
    eye_points format: [p1, p2, p3, p4, p5, p6]
    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    """
    v1 = compute_distance(eye_points[1], eye_points[5])
    v2 = compute_distance(eye_points[2], eye_points[4])
    h = compute_distance(eye_points[0], eye_points[3])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)

def run_detection(camera_index=0, ear_threshold=0.22, consecutive_frames=15):
    """
    Real-time video capture loop monitoring driver eye state.
    Supports OpenCV Haar Cascade fallback if Dlib predictor is unavailable.
    """
    print("Initializing video stream...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Unable to access camera at index {camera_index}")
        return

    # Load face and eye cascades with verification
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    if face_cascade.empty() or eye_cascade.empty():
        raise RuntimeError("Failed to load OpenCV Haar Cascade XML models from cv2.data.haarcascades.")

    sleep_counter = 0
    missing_face_counter = 0
    drowsy_counter = 0
    active_counter = 0
    status = "Driver Status: Alert"
    color = (0, 255, 0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

            if len(faces) == 0:
                missing_face_counter += 1
                if missing_face_counter > consecutive_frames:
                    status = "WARNING: NO FACE DETECTED"
                    color = (0, 0, 255)
            else:
                missing_face_counter = 0

                # Select primary driver face (largest detected bounding box) to prevent counter inflation
                driver_face = max(faces, key=lambda b: b[2] * b[3])
                for (fx, fy, fw, fh) in faces:
                    is_primary = (fx, fy, fw, fh) == tuple(driver_face)
                    box_color = (0, 255, 255) if is_primary else (200, 200, 200)
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), box_color, 2)

                x, y, w, h = driver_face
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))

                if len(eyes) == 0:
                    sleep_counter += 1
                    active_counter = 0
                else:
                    active_counter += 1
                    sleep_counter = max(0, sleep_counter - 1)

                if sleep_counter > consecutive_frames:
                    status = "WARNING: DROWSINESS DETECTED"
                    color = (0, 0, 255)
                elif sleep_counter > 5:
                    status = "Caution: Eyes Closing"
                    color = (0, 165, 255)
                else:
                    status = "Driver Status: Alert"
                    color = (0, 255, 0)

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

            cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Closed Frame Counter: {sleep_counter}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            cv2.putText(frame, "Press 'Q' or 'ESC' to exit", (30, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            cv2.imshow("Driver Fatigue Monitoring System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera stream closed.")

if __name__ == "__main__":
    run_detection()
