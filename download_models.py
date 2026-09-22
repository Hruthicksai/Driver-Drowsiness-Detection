import os
import sys
import urllib.request
import time
import bz2
import shutil

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

LANDMARK_SOURCES = [
    "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2",
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    "https://huggingface.co/matt3ounstable/dlib_predictor_recognition/resolve/main/shape_predictor_68_face_landmarks.dat"
]

YOLO_FILES = {
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


def download_file(url, dest_path):
    print(f"\n[*] Downloading {os.path.basename(dest_path)}...", flush=True)
    print(f"    Source: {url}", flush=True)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )

    temp_dest = dest_path + ".tmp"
    with urllib.request.urlopen(req, timeout=45) as response, open(temp_dest, "wb") as out_file:
        total_size = response.getheader("Content-Length")
        total_size = int(total_size) if total_size else None
        downloaded = 0
        chunk_size = 1024 * 512  # 512 KB
        last_print = time.time()

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)

            if time.time() - last_print > 0.3:
                last_print = time.time()
                if total_size:
                    pct = (downloaded / total_size) * 100
                    mb_cur = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    print(f"\r    Progress: {mb_cur:.1f} MB / {mb_tot:.1f} MB ({pct:.1f}%)", end="", flush=True)
                else:
                    mb_cur = downloaded / (1024 * 1024)
                    print(f"\r    Downloaded: {mb_cur:.1f} MB", end="", flush=True)

    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename(temp_dest, dest_path)
    print(f"\n    [+] Saved: {os.path.basename(dest_path)} ({os.path.getsize(dest_path) / (1024 * 1024):.1f} MB)", flush=True)


def setup_landmarks():
    target_dat = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
    # Valid extracted file is ~99MB
    if os.path.exists(target_dat) and os.path.getsize(target_dat) > 90_000_000:
        print(f"[OK] shape_predictor_68_face_landmarks.dat already present ({os.path.getsize(target_dat) / (1024*1024):.1f} MB).", flush=True)
        return True

    for url in LANDMARK_SOURCES:
        try:
            if url.endswith(".bz2"):
                bz2_file = target_dat + ".bz2"
                download_file(url, bz2_file)
                print(f"[*] Extracting {os.path.basename(bz2_file)}...", flush=True)
                with bz2.BZ2File(bz2_file, "rb") as src, open(target_dat, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                try:
                    os.remove(bz2_file)
                except OSError:
                    pass
                print(f"    [+] Extracted shape_predictor_68_face_landmarks.dat ({os.path.getsize(target_dat) / (1024*1024):.1f} MB)", flush=True)
                return True
            else:
                download_file(url, target_dat)
                return True
        except Exception as e:
            print(f"\n    [!] Failed from {url}: {e}", flush=True)

    return False


def setup_yolo():
    for filename, urls in YOLO_FILES.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 100:
            print(f"[OK] {filename} already present.", flush=True)
            continue

        downloaded = False
        for url in urls:
            try:
                download_file(url, dest)
                downloaded = True
                break
            except Exception as e:
                print(f"\n    [!] Failed to download {filename} from {url}: {e}", flush=True)

        if not downloaded:
            print(f"[!] Warning: Could not download {filename}.", flush=True)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 60, flush=True)
    print(" Driver Monitoring System - Model Setup", flush=True)
    print("=" * 60, flush=True)

    # 1. Landmarks
    success = setup_landmarks()
    if not success:
        print("[!] Failed to setup face landmark model.", flush=True)

    # 2. YOLO
    setup_yolo()

    print("\n" + "=" * 60, flush=True)
    print(" Setup finished! Run 'python main.py' to launch.", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
