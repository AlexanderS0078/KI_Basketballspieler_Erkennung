from typing import Generator
import time, cv2, numpy as np
from fastapi.responses import StreamingResponse
from nicegui import ui, app
from ultralytics import YOLO

# ---- set your local path here ----
VIDEO_PATH = 'A:\downloads2/ball.mp4'
MODEL_PATH = 'A:\downloads2/best.pt'
CONF = 0.50

model = YOLO(MODEL_PATH)

# pre-warm so first frame doesn't block for seconds
_ = model.predict(np.zeros((640, 640, 3), dtype=np.uint8), conf=CONF, verbose=False)

DETECT_ON = True
ERROR_TEXT = ''


def draw(result, frame):
    names = result.names
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0]); conf = float(box.conf[0])
        label = f'{names[cls]} {conf:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        t, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - t[1] - 6), (x1 + t[0] + 4, y1), (0,255,0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    return frame


def mjpeg() -> Generator[bytes, None, None]:
    global ERROR_TEXT
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        ERROR_TEXT = f'Could not open: {VIDEO_PATH}'
        yield _error_frame(ERROR_TEXT)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    sleep_s = max(0.001, 1.0 / fps)

    while True:
        ok, frame = cap.read()
        if not ok:
            # loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # (optional) downscale huge 4K videos for speed
        if frame.shape[1] > 1920:
            frame = cv2.resize(frame, (1280, int(1280 * frame.shape[0] / frame.shape[1])))

        try:
            if DETECT_ON:
                t0 = time.time()
                res = model(frame, conf=CONF, verbose=False)[0]
                frame = draw(res, frame)
                # if inference takes too long, we still yield the frame quickly next loop
                # (no blocking inside the generator)
                _ = t0
        except Exception as e:
            ERROR_TEXT = f'YOLO error: {e!s}'
            yield _error_frame(ERROR_TEXT)
            return

        ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + jpg.tobytes() + b'\r\n')

        time.sleep(sleep_s)


def _error_frame(text: str) -> bytes:
    """Yield a simple JPEG with the error text so you see it in the browser."""
    img = np.full((320, 960, 3), 30, np.uint8)
    cv2.putText(img, text, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    ok, jpg = cv2.imencode('.jpg', img)
    body = jpg.tobytes() if ok else b''
    return (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + body + b'\r\n')


@app.get('/yolo_stream')
def yolo_stream():
    return StreamingResponse(mjpeg(), media_type='multipart/x-mixed-replace; boundary=frame')


@ui.page('/')
def index():
    ui.markdown('### Local MP4 → YOLOv8 (realtime boxes)')
    ui.label(f'Video: {VIDEO_PATH}')
    with ui.row().classes('items-center gap-3'):
        def toggle(e):
            global DETECT_ON
            DETECT_ON = e.value
        ui.switch('Detection on', value=True, on_change=toggle)
        def set_conf(e):
            global CONF
            CONF = e.value
        ui.slider(min=0.1, max=0.8, value=CONF, step=0.05,
                  on_change=set_conf).props('label-always')
    ui.image('/yolo_stream').classes('w-full max-w-4xl rounded-xl shadow mt-2')
    if ERROR_TEXT:
        ui.label(ERROR_TEXT).classes('text-red-500')

ui.run(port = 8089, title='YOLOv8 + NiceGUI', reload=False)