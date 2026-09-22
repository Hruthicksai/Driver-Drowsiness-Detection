import os
import sys
import cv2
import dlib
from imutils import face_utils

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FACE_LANDMARK_MODEL,
    EAR_THRESHOLD,
    EAR_CONSEC_FRAMES,
    BLINK_MIN_FRAMES,
    BLINK_MAX_FRAMES,
    MAR_THRESHOLD,
    YAWN_CONSEC_FRAMES,
    MICROSLEEP_FRAMES,
    HEAD_TILT_THRESHOLD,
    PERCLOS_WINDOW,
    PERCLOS_THRESHOLD,
    YOLO_CONFIG,
    YOLO_WEIGHTS,
    YOLO_CLASSES
)
from detection_utils import (
    eye_aspect_ratio,
    mouth_aspect_ratio,
    get_head_pose_angle
)
from alert_system import AlertSystem, visual_alert
from phone_detector import PhoneDetector


if not os.path.exists(FACE_LANDMARK_MODEL):
    print("\n" + "=" * 65)
    print(" [ERROR] Face landmark model not found!")
    print(f" Expected location: {FACE_LANDMARK_MODEL}")
    print(" Please run the following command to download the model files:")
    print("     python download_models.py")
    print("=" * 65 + "\n")
    sys.exit(1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(FACE_LANDMARK_MODEL)

(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

phone_detector = PhoneDetector(
    YOLO_CONFIG,
    YOLO_WEIGHTS,
    YOLO_CLASSES
)

alert_system = AlertSystem()

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"\n[ERROR] Could not access webcam at index {CAMERA_INDEX}.")
    print("Please verify that your camera is connected and not being used by another application.\n")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

eye_closed_frames = 0
yawn_frames = 0
blink_count = 0
yawn_count = 0

perclos_history = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    drowsy = False
    yawning = False
    head_tilt = False

    for rect in faces:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # -----------------------------
        # EYE DETECTION
        # -----------------------------

        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        cv2.polylines(
            frame,
            [left_eye],
            True,
            (255, 255, 0),
            1
        )

        cv2.polylines(
            frame,
            [right_eye],
            True,
            (255, 255, 0),
            1
        )

        if ear < EAR_THRESHOLD:
            eye_closed_frames += 1
        else:

            if BLINK_MIN_FRAMES <= eye_closed_frames <= BLINK_MAX_FRAMES:
                blink_count += 1

            eye_closed_frames = 0

        if eye_closed_frames >= EAR_CONSEC_FRAMES:
            drowsy = True

        if eye_closed_frames >= MICROSLEEP_FRAMES:
            drowsy = True

        # -----------------------------
        # MOUTH / YAWN DETECTION
        # -----------------------------

        mouth = shape[mouth_start:mouth_end]

        mar = mouth_aspect_ratio(mouth)

        cv2.polylines(
            frame,
            [mouth],
            True,
            (255, 0, 255),
            1
        )

        if mar > MAR_THRESHOLD:
            yawn_frames += 1
        else:

            if yawn_frames >= YAWN_CONSEC_FRAMES:
                yawn_count += 1

            yawn_frames = 0

        if yawn_frames >= YAWN_CONSEC_FRAMES:
            yawning = True

        # -----------------------------
        # HEAD POSE
        # -----------------------------

        pitch, yaw, roll = get_head_pose_angle(
            shape,
            frame
        )

        if abs(roll) > HEAD_TILT_THRESHOLD:
            head_tilt = True

        # -----------------------------
        # PERCLOS
        # -----------------------------

        perclos_history.append(
            1 if ear < EAR_THRESHOLD else 0
        )

        if len(perclos_history) > PERCLOS_WINDOW:
            perclos_history.pop(0)

        perclos = (
            sum(perclos_history) /
            len(perclos_history)
        )

        if perclos > PERCLOS_THRESHOLD:
            drowsy = True

        # -----------------------------
        # DISPLAY
        # -----------------------------

        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"MAR: {mar:.2f}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Blinks: {blink_count}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Yawns: {yawn_count}",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"PERCLOS: {perclos:.2f}",
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # -----------------------------
    # PHONE DETECTION
    # -----------------------------

    phone_detected = phone_detector.detect(frame)

    # -----------------------------
    # ALERTS
    # -----------------------------

    if phone_detected:

        visual_alert(
            frame,
            "PHONE USAGE DETECTED!"
        )

        alert_system.alert("PHONE")

    elif drowsy:

        visual_alert(
            frame,
            "DROWSINESS DETECTED!"
        )

        alert_system.alert("DROWSINESS")

    elif yawning:

        visual_alert(
            frame,
            "YAWNING DETECTED!"
        )

        alert_system.alert("YAWNING")

    elif head_tilt:

        visual_alert(
            frame,
            "HEAD TILT DETECTED!"
        )

        alert_system.alert("HEAD TILT")

    else:

        cv2.putText(
            frame,
            "DRIVER STATUS: ALERT",
            (20, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Driver Monitoring System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()