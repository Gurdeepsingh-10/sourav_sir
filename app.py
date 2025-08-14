import cv2
import threading
import time
import random
from flask import Flask, render_template, Response, request
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# --------- User-configurable defaults (keeps same param names) ----------
MODEL_PATH = "yolov8n.pt"   # change to yolov8s.pt or yolov8m.pt if you want more accuracy
FRAME_SKIP = 3             # detector will sample every N frames (adjustable via UI)
JPEG_QUALITY = 80
TARGET_DETECT_SIZE = 480    # imgsz used for model inference (keeps input small)
# ----------------------------------------------------------------------

# Parameter state (adjustable by UI)
current_confidence = 0.5
current_distance_cap = 50
frame_skip = FRAME_SKIP

# Thread-shared frames/state
latest_frame = None         # raw latest BGR frame from camera
annotated_frame = None      # latest frame annotated by detector to stream
latest_frame_ts = 0.0
lock = threading.Lock()

# class color mapping
class_colors = {}
random.seed(42)
def get_class_color(cls_id):
    if cls_id not in class_colors:
        class_colors[cls_id] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    return class_colors[cls_id]

# --------- Load model (attempt GPU fp16 if available) ----------
print("[INFO] Loading model:", MODEL_PATH)
model = YOLO(MODEL_PATH)

# Try to use CUDA fp16 if available (ultralytics supports model.to or model.fuse)
try:
    import torch
    if torch.cuda.is_available():
        model.to("cuda")
        # ultralytics automatically uses best dtype; optionally we can set half precision for speed
        try:
            model.model.half()  # best-effort, may not be available for every model wrapper
            print("[INFO] Using CUDA + FP16 for inference")
        except Exception:
            print("[INFO] CUDA available but couldn't force half(); still on CUDA.")
    else:
        print("[INFO] CUDA not available — running on CPU")
except Exception:
    print("[INFO] torch not available or error checking cuda — default model device used")

# --------- Camera capture thread (continually updates latest_frame) ----------
def camera_capture(device_index=0):
    global latest_frame, latest_frame_ts
    # Use CAP_DSHOW on Windows for lower latency; fallback automatically on other OS
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    # Optionally set a capture resolution (you can reduce if too slow)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # try to avoid internal buffering

    if not cap.isOpened():
        print("[ERROR] Camera not opened. Check device index.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # remove mirror if you previously wanted a natural view (flip=1) or keep unflipped if you want
        frame = cv2.flip(frame, 1)  # keep same behavior as before

        # update latest frame (non-blocking short lock)
        with lock:
            latest_frame = frame
            latest_frame_ts = time.time()

        # tiny sleep to yield CPU (capture runs as fast as camera allows)
        time.sleep(0.005)

# --------- Detector thread (samples latest_frame every frame_skip frames) ----------
def detection_loop():
    global annotated_frame, latest_frame, frame_skip, current_confidence, current_distance_cap
    frame_counter = 0
    last_detection = None

    # Pre-calc letterbox target if needed later; we'll keep inference on resized image
    while True:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        frame_counter += 1

        # Run detection only every N frames (N = frame_skip) to save CPU
        if frame_counter % max(1, frame_skip) == 0:
            # Prepare small image for detections (maintain aspect ratio by resizing shortest edge)
            h, w = frame.shape[:2]
            # scale so that the longer side is NOT bigger than TARGET_DETECT_SIZE while preserving aspect
            # but to keep consistent scaling, we scale by largest dimension -> use imgsz param below instead
            try:
                # Ultralytics model inference with imgsz parameter (fast)
                results = model(frame, conf=current_confidence, imgsz=TARGET_DETECT_SIZE)
            except TypeError:
                # fallback if imgsz not supported: manual resize
                scale = TARGET_DETECT_SIZE / max(h, w)
                small = cv2.resize(frame, (int(w*scale), int(h*scale)))
                results = model(small, conf=current_confidence)

            # Build annotated copy from original-size frame so boxes align with stream
            annotated = frame.copy()

            # ulralytics returns .boxes on results[0] when not streaming; handle both types
            # Results can be a Results object or a list; normalize:
            try:
                res_list = results if isinstance(results, list) else [results]
            except Exception:
                res_list = [results]

            # Use the first results object for typical single-image inference
            r0 = res_list[0]
            # iterate boxes if present
            boxes = getattr(r0, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    # box.xyxy may be a Tensor; convert
                    try:
                        xyxy = box.xyxy[0].cpu().numpy()
                    except Exception:
                        xyxy = np.array(box.xyxy[0])

                    x1, y1, x2, y2 = map(int, xyxy)
                    bbox_height = y2 - y1
                    try:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                    except Exception:
                        # fallback if fields differ
                        conf = float(box.conf) if hasattr(box, "conf") else 0.0
                        cls_id = int(box.cls) if hasattr(box, "cls") else 0

                    # distance cap check (approx by bbox height in pixels)
                    if bbox_height < current_distance_cap:
                        continue

                    name = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
                    color = get_class_color(cls_id)
                    label = f"{name} {conf:.2f}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # commit annotated frame
            with lock:
                annotated_frame = annotated
                # small timestamp to indicate detection time (optional)
                annotated_frame_ts = time.time()

        # small sleep to avoid 100% CPU while waiting next sampling opportunity
        time.sleep(0.002)


# --------- Streaming generator (non-blocking; always returns latest annotated if available) ----------
def generate_frames():
    global annotated_frame, latest_frame
    while True:
        with lock:
            frame = annotated_frame if annotated_frame is not None else latest_frame

        if frame is None:
            # no frame yet; wait briefly
            time.sleep(0.01)
            continue

        # encode
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ret:
            time.sleep(0.005)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --------- Flask routes (same endpoints as before; UI doesn't need changes) ----------
@app.route('/')
def index():
    # pass current values so your template can show them if desired
    return render_template('index.html', confidence=current_confidence, distance=current_distance_cap, frame_skip=frame_skip)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_params', methods=['POST'])
def set_params():
    global current_confidence, current_distance_cap, frame_skip
    try:
        current_confidence = float(request.form.get("confidence", current_confidence))
        current_distance_cap = int(request.form.get("distance", current_distance_cap))
        frame_skip = int(request.form.get("frame_skip", frame_skip))
        # clamp sensible ranges
        current_confidence = min(max(current_confidence, 0.01), 0.99)
        current_distance_cap = max(0, current_distance_cap)
        frame_skip = max(1, frame_skip)
        return "OK", 200
    except Exception as e:
        print("[ERROR] set_params:", e)
        return "ERR", 400

# --------- Start threads and run ----------
if __name__ == "__main__":
    # start camera and detector threads
    t_cam = threading.Thread(target=camera_capture, args=(0,), daemon=True)
    t_det = threading.Thread(target=detection_loop, daemon=True)
    t_cam.start()
    t_det.start()

    # run Flask app (no debug; threaded True is fine)
    app.run(host="0.0.0.0", port=5000)

