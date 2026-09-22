import os
import sys
import urllib.request
import time

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODEL_FILES = {
    "shape_predictor_68_face_landmarks.dat": [
        "https://huggingface.co/matt3ounstable/dlib_predictor_recognition/resolve/main/shape_predictor_68_face_landmarks.dat",
        "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
    ],
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
    with urllib.request.urlopen(req, timeout=30) as response, open(temp_dest, "wb") as out_file:
        total_size = response.getheader("Content-Length")
        total_size = int(total_size) if total_size else None
        downloaded = 0
        chunk_size = 1024 * 512  # 512 KB chunks
        last_print = time.time()

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)

            # Throttle console printing to every 0.3s
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

    if os.path.exists(temp_dest):
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(temp_dest, dest_path)
    print(f"\n    [+] Successfully saved {os.path.basename(dest_path)}", flush=True)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 60, flush=True)
    print(" Driver Monitoring System - Model Downloader", flush=True)
    print("=" * 60, flush=True)

    for filename, urls in MODEL_FILES.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"[OK] {filename} already present ({size_mb:.1f} MB).", flush=True)
            continue

        downloaded = False
        for url in urls:
            try:
                download_file(url, dest)
                downloaded = True
                break
            except Exception as e:
                print(f"\n    [!] Download failed from {url}: {e}", flush=True)

        if not downloaded:
            print(f"[!] Warning: Could not download {filename}.", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(" Model setup complete! You can now run 'python main.py'", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
