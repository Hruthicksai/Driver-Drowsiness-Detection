import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# -----------------------------
# MODEL PATHS
# -----------------------------
FACE_LANDMARK_MODEL = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")

YOLO_CONFIG = os.path.join(MODELS_DIR, "yolov4-tiny.cfg")
YOLO_WEIGHTS = os.path.join(MODELS_DIR, "yolov4-tiny.weights")
YOLO_CLASSES = os.path.join(MODELS_DIR, "coco.names")


# -----------------------------
# CAMERA
# -----------------------------
CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


# -----------------------------
# EYE ASPECT RATIO
# -----------------------------
EAR_THRESHOLD = 0.22

# Number of consecutive frames
# with low EAR before drowsiness alert
EAR_CONSEC_FRAMES = 20


# -----------------------------
# MOUTH ASPECT RATIO
# -----------------------------
MAR_THRESHOLD = 0.65

# Number of consecutive yawning frames
YAWN_CONSEC_FRAMES = 15


# -----------------------------
# BLINK DETECTION
# -----------------------------
BLINK_MIN_FRAMES = 2
BLINK_MAX_FRAMES = 15


# -----------------------------
# MICROSLEEP
# -----------------------------
MICROSLEEP_FRAMES = 40


# -----------------------------
# HEAD TILT
# -----------------------------
HEAD_TILT_THRESHOLD = 25


# -----------------------------
# PERCLOS
# -----------------------------
PERCLOS_WINDOW = 150

PERCLOS_THRESHOLD = 0.40


# -----------------------------
# YOLO
# -----------------------------
YOLO_CONFIDENCE = 0.5
YOLO_NMS_THRESHOLD = 0.4

PHONE_CLASS_NAME = "cell phone"


# -----------------------------
# ALERT
# -----------------------------
ALERT_COOLDOWN = 3