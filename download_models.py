import os
import sys
import urllib.request
import bz2
import shutil

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODEL_FILES = {
    "coco.names": [
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
    ],
    "yolov4-tiny.cfg": [
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
    ],
    "yolov4-tiny.weights": [
        "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
    ]
}

DLIB_LANDMARK_URLS = [
    "https://raw.githubusercontent.com/davisking/dlib-models/master/shape_predictor_68_face_landmarks.dat.bz2",
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
]


def download_with_progress(url, dest_path):
    print(f"Downloading {os.path.basename(dest_path)} from:\n  {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_size = response.getheader("Content-Length")
        total_size = int(total_size) if total_size else None
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = downloaded / total_size * 100
                mb_down = downloaded / (1024 * 1024)
                mb_tot = total_size / (1024 * 1024)
                print(f"\r  Progress: {mb_down:.1f}MB / {mb_tot:.1f}MB ({percent:.1f}%)", end="")
            else:
                mb_down = downloaded / (1024 * 1024)
                print(f"\r  Downloaded: {mb_down:.1f}MB", end="")
        print("\n  [Done]")


def setup_dlib_landmarks():
    dat_target = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
    if os.path.exists(dat_target) and os.path.getsize(dat_target) > 10_000_000:
        print("[OK] shape_predictor_68_face_landmarks.dat already exists.")
        return

    bz2_target = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat.bz2")
    downloaded = False
    for url in DLIB_LANDMARK_URLS:
        try:
            download_with_progress(url, bz2_target)
            downloaded = True
            break
        except Exception as e:
            print(f"  Failed with {url}: {e}. Trying next source...")

    if not downloaded:
        print("[ERROR] Could not download shape_predictor_68_face_landmarks.dat.bz2.")
        return

    print("Decompressing shape_predictor_68_face_landmarks.dat.bz2...")
    with bz2.BZ2File(bz2_target, "rb") as source, open(dat_target, "wb") as dest:
        shutil.copyfileobj(source, dest)

    # Clean up bz2 archive
    try:
        os.remove(bz2_target)
    except OSError:
        pass
    print("  [Done] Extracted shape_predictor_68_face_landmarks.dat")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 60)
    print("Driver Monitoring System - Model Downloader")
    print("=" * 60)

    # 1. dlib landmarks
    setup_dlib_landmarks()

    # 2. YOLO files
    for filename, urls in MODEL_FILES.items():
        target = os.path.join(MODELS_DIR, filename)
        if os.path.exists(target) and os.path.getsize(target) > 0:
            print(f"[OK] {filename} already exists.")
            continue

        for url in urls:
            try:
                download_with_progress(url, target)
                break
            except Exception as e:
                print(f"  Failed with {url}: {e}. Trying next source...")

    print("\nAll model checks completed!")


if __name__ == "__main__":
    main()
