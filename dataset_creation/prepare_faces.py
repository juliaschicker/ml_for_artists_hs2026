# Prepare a face dataset: find every face in a folder of images and videos,
# cut out a square with the face in the centre, and give all crops the same size.
#
# How it works:
#   1. Goes through INPUT_DIR: images (incl. iPhone HEIC) are read directly, from videos every FRAME_STEP-th frame is used.
#   2. Face detection with landmarks (OpenCV's YuNet detector): for every face it finds a box
#      plus 5 landmarks -- both eyes, the nose tip and both mouth corners.
#   3. For every face, a square around the face centre is cut out (optionally rotated so the eyes are level).
#   4. At the end, all crops are resized to OUTPUT_SIZE x OUTPUT_SIZE pixels.
#
# Run from this folder:   python prepare_faces.py

import math
import urllib.request
from pathlib import Path

# OpenCV: reads images and videos, detects faces, crops and resizes
import cv2
# numpy: number arrays (OpenCV stores images as numpy arrays)
import numpy as np
# Pillow: used only for HEIC/HEIF photos (iPhone format), which OpenCV can't read
from PIL import Image, ImageOps

# pillow-heif teaches Pillow to open HEIC files (install with: python -m pip install pillow-heif)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False


# ----- Settings -----
INPUT_DIR = "input"          # folder with your images and videos
OUTPUT_DIR = "faces"         # folder the face crops are saved to (created automatically)
RECURSIVE = True             # also look into subfolders of INPUT_DIR

FRAME_STEP = 10              # videos: use every 10th frame
OUTPUT_SIZE = 256            # final size of every crop in pixels (e.g. 64, 128, 256, 512)

CROP_SCALE = 2.0             # size of the square relative to the face: 1.2 = tight face, 2.0 = whole head, 3.0 = head + shoulders
ALIGN_EYES = True            # rotate each crop so the eyes are horizontal
EDGE_MODE = "replicate"      # face near the image border: "black" (fill with black), "replicate" (stretch edge pixels),
                             # "mirror" (mirror the image), or "skip" (leave out faces whose square doesn't fit)

MIN_CONFIDENCE = 0.8         # 0...1: how sure the detector must be that it's a face (higher = fewer false faces)
MIN_FACE_SIZE = 40           # ignore faces smaller than this (in pixels, original image); tiny faces get blurry when enlarged
DETECTION_MAX_SIDE = 1280    # big images are scaled down to this size for detection only (faster; crops use full quality)

JPEG_QUALITY = 95

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
HEIC_TYPES = {".heic", ".heif"}  # iPhone photos; need the pillow-heif package
VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

# Face detection model (small, ~230 KB) -- downloaded automatically the first time
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL_PATH = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"


# ----- Setup -----
def load_detector():
    """Load the YuNet face detector, downloading the model file if needed."""
    if not MODEL_PATH.exists():
        print("Downloading face detection model ...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (320, 320), MIN_CONFIDENCE)


BORDER = {"black": cv2.BORDER_CONSTANT, "replicate": cv2.BORDER_REPLICATE,
          "mirror": cv2.BORDER_REFLECT, "skip": cv2.BORDER_CONSTANT}


# ----- Reading images and video frames -----
def iterate_sources(input_dir):
    """Yields (name, image) for every image and every FRAME_STEP-th video frame."""
    pattern = "**/*" if RECURSIVE else "*"
    for path in sorted(Path(input_dir).glob(pattern)):
        suffix = path.suffix.lower()
        if suffix in IMAGE_TYPES:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)   # also applies the phone's rotation info (EXIF)
            if image is None:
                print(f"  could not read {path.name}, skipped")
                continue
            yield path.stem, image
        elif suffix in HEIC_TYPES:
            if not HEIC_SUPPORT:
                print(f"  {path.name} skipped: HEIC needs the pillow-heif package (python -m pip install pillow-heif)")
                continue
            with Image.open(path) as heic:
                heic = ImageOps.exif_transpose(heic).convert("RGB")      # apply the phone's rotation info
                image = cv2.cvtColor(np.array(heic), cv2.COLOR_RGB2BGR)  # Pillow uses RGB, OpenCV uses BGR
            yield path.stem, image
        elif suffix in VIDEO_TYPES:
            video = cv2.VideoCapture(str(path))
            frame_index = 0
            while True:
                ok, frame = video.read()
                if not ok:
                    break
                if frame_index % FRAME_STEP == 0:
                    yield f"{path.stem}_frame{frame_index:06d}", frame
                frame_index += 1
            video.release()


# ----- Face detection -----
def detect_faces(detector, image):
    """Returns a list of faces: (box_x, box_y, box_w, box_h, eye_1, eye_2, confidence) in original image pixels."""
    h, w = image.shape[:2]
    scale = min(1.0, DETECTION_MAX_SIDE / max(h, w))
    small = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = detector.detect(small)
    if faces is None:
        return []

    results = []
    for f in faces:
        f = f[:14] / scale   # back to original image coordinates (index 14 is the confidence, not a coordinate)
        x, y, fw, fh = f[0:4]
        eye_1, eye_2 = (f[4], f[5]), (f[6], f[7])   # YuNet's 5 landmarks: eye, eye, nose tip, mouth corner, mouth corner
        if max(fw, fh) < MIN_FACE_SIZE:
            continue
        results.append((x, y, fw, fh, eye_1, eye_2))
    return results


# ----- Cutting the square -----
def crop_face(image, face):
    """Cuts a square around the face (rotated so the eyes are level if ALIGN_EYES). Returns None if it doesn't fit and EDGE_MODE is 'skip'."""
    x, y, fw, fh, eye_1, eye_2 = face
    centre_x, centre_y = x + fw / 2, y + fh / 2
    side = int(round(max(fw, fh) * CROP_SCALE))

    angle = 0.0
    if ALIGN_EYES:
        angle = math.degrees(math.atan2(eye_2[1] - eye_1[1], eye_2[0] - eye_1[0]))

    # One transformation that rotates around the face centre and moves the centre into the middle of the square
    matrix = cv2.getRotationMatrix2D((centre_x, centre_y), angle, 1.0)
    matrix[0, 2] += side / 2 - centre_x
    matrix[1, 2] += side / 2 - centre_y

    if EDGE_MODE == "skip":
        # Check whether all 4 corners of the square lie inside the original image
        inverse = cv2.invertAffineTransform(matrix)
        corners = np.array([[0, 0, 1], [side, 0, 1], [0, side, 1], [side, side, 1]], dtype=np.float64)
        original = corners @ inverse.T
        h, w = image.shape[:2]
        if (original[:, 0].min() < 0 or original[:, 1].min() < 0 or
                original[:, 0].max() > w or original[:, 1].max() > h):
            return None

    return cv2.warpAffine(image, matrix, (side, side), flags=cv2.INTER_CUBIC, borderMode=BORDER[EDGE_MODE])


# ----- Unifying the sizes -----
def resize_all(paths):
    """Resize every crop made in this run to OUTPUT_SIZE x OUTPUT_SIZE."""
    upscaled = 0
    for path in paths:
        image = cv2.imread(str(path))
        if image.shape[0] < OUTPUT_SIZE:
            upscaled += 1
            interpolation = cv2.INTER_CUBIC   # enlarging
        else:
            interpolation = cv2.INTER_AREA    # shrinking (keeps it sharp without flicker)
        image = cv2.resize(image, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=interpolation)
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return upscaled


def main():
    base = Path(__file__).parent
    input_dir, output_dir = base / INPUT_DIR, base / OUTPUT_DIR
    if not input_dir.exists():
        print(f"Input folder not found: {input_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = load_detector()
    saved, sources, skipped_edge = [], 0, 0

    print(f"Looking for faces in {input_dir} ...")
    for name, image in iterate_sources(input_dir):
        sources += 1
        faces = detect_faces(detector, image)
        for n, face in enumerate(faces):
            crop = crop_face(image, face)
            if crop is None:
                skipped_edge += 1
                continue
            out_path = output_dir / f"{name}_face{n}.jpg"
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            saved.append(out_path)
        if sources % 50 == 0:
            print(f"  {sources} images/frames checked, {len(saved)} face crops so far")

    print(f"\nChecked {sources} images/frames, saved {len(saved)} face crops.")
    if skipped_edge:
        print(f"  {skipped_edge} faces skipped because they were too close to the image border (EDGE_MODE = 'skip').")

    print(f"Resizing all crops to {OUTPUT_SIZE} x {OUTPUT_SIZE} ...")
    upscaled = resize_all(saved)
    if upscaled:
        print(f"  note: {upscaled} crops were smaller than {OUTPUT_SIZE}px and had to be enlarged (may look soft)."
              f" Raise MIN_FACE_SIZE to leave those out.")
    print(f"Done. Faces saved in {output_dir}")


if __name__ == "__main__":
    main()
