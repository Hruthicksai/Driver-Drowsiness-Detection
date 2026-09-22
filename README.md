# Driver Monitoring System (DMS)

An intelligent real-time Driver Monitoring and Drowsiness Detection System built using Python, OpenCV, and dlib. The system monitors the driver's face via webcam to detect signs of fatigue, sleepiness, distraction, and mobile phone usage.

---

## 🚀 Features

- **Eye Aspect Ratio (EAR) & PERCLOS**: Real-time eye closure analysis to detect drowsiness and microsleep episodes.
- **Mouth Aspect Ratio (MAR)**: Yawn frequency and prolonged yawning detection.
- **Head Pose Estimation**: 3D orientation analysis (pitch, yaw, roll) using solvePnP to flag head tilts or inattention.
- **Blink & Yawn Counters**: Real-time counter displayed directly on the heads-up interface.
- **Distraction & Phone Detection**: YOLO-based object detection for recognizing cell phone use while driving.
- **Multimodal Alerts**: Visual warnings on screen along with audio beeps (`winsound`).

---

## 🛠️ Requirements & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Hruthicksai/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
> **Note for Windows users**: `requirements.txt` uses `dlib-bin` to provide pre-compiled wheels, avoiding the need for CMake and Visual C++ Build Tools.

### 4. Download Pretrained Models
Run the automated downloader script to fetch the 68-point facial landmark model and YOLO weights:
```bash
python download_models.py
```
This will place the following files into the `models/` directory:
- `shape_predictor_68_face_landmarks.dat`
- `yolov4-tiny.weights`
- `yolov4-tiny.cfg`
- `coco.names`

---

## 💻 Running the Project in VS Code

1. Open VS Code in the project folder:
   ```bash
   code .
   ```
2. Select your Python interpreter:
   - Press `Ctrl + Shift + P` -> `Python: Select Interpreter`
   - Choose your Python environment.
3. Open `main.py` and run it:
   - Press `F5` (or click the Run button in the top right).
   - Alternatively, open the built-in terminal and run:
     ```bash
     python main.py
     ```
4. Press `q` while focused on the video window to safely exit.

---

## 📁 Project Structure

```
Driver-Monitoring-System/
│
├── models/                     # Model weights and configs (auto-downloaded)
│   └── .gitkeep
├── alert_system.py             # Audio and visual alert manager
├── config.py                   # Central configuration & thresholds
├── detection_utils.py          # Geometric math (EAR, MAR, solvePnP Head Pose)
├── download_models.py          # Automatic model downloader
├── main.py                     # Main application loop & video pipeline
├── phone_detector.py           # YOLO-based cell phone detector
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

---

## ⚙️ Configuration & Thresholds

All thresholds and parameters can be adjusted in `config.py`:
- `EAR_THRESHOLD`: Eye closure threshold (default: `0.22`)
- `EAR_CONSEC_FRAMES`: Consecutive frames of closed eyes to trigger drowsiness alert (default: `20`)
- `MAR_THRESHOLD`: Mouth openness threshold for yawn detection (default: `0.65`)
- `HEAD_TILT_THRESHOLD`: Maximum head tilt roll angle before alerting (default: `25°`)
- `PERCLOS_THRESHOLD`: Percentage of eye closure threshold (default: `0.40`)
- `CAMERA_INDEX`: Camera device index (default: `0`)
