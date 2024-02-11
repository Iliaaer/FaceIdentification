import threading
import cv2
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, StreamingResponse

from RetinaFace.mobile_net import Net

cap = cv2.VideoCapture(0)
net = Net(load2cpu=False)

outputFrame1: cv2.typing.MatLike = None
outputFrame2: cv2.typing.MatLike = None
lock = threading.Lock()

router = APIRouter(
    prefix="/camera",
    tags=["Camera"]
)

window: bool = False

H, W = None, None


def get_frame():
    """Функция генератора потокового видео."""
    global outputFrame1, lock
    global cap
    global H, W
    print(0)
    while True:
        if not cap.isOpened():
            break
        ret, frame = cap.read()
        if not H:
            H, W = frame.shape[:2]
        if not ret:
            continue
        if window:
            cv2.imshow("1", frame)
            key = cv2.waitKey(1) & 0xff
        with lock:
            outputFrame1 = frame.copy()
    if window:
        cv2.destroyWindow("1")
    print("EXIT get_frame")


def get_frame_face():
    global outputFrame1, lock
    global outputFrame2
    global net, cap
    global H, W
    print(1)
    while True:
        if not cap.isOpened():
            break
        if net is None:
            continue
        with lock:
            if outputFrame1 is None:
                continue
            frame = outputFrame1.copy()

        if frame is None:
            continue

        dets = net.detect(frame)

        for b in dets:
            if b[4] < 0.7:
                continue
            b_new = list(map(int, b))
            x1, y1, x2, y2 = b_new[:4]

            x1 = max(0, min(W, x1))
            x2 = max(0, min(W, x2))
            y1 = max(0, min(H, y1))
            y2 = max(0, min(H, y2))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        if window:
            cv2.imshow("2", frame)
            key = cv2.waitKey(1) & 0xff

        with lock:
            outputFrame2 = frame.copy()
    if window:
        cv2.destroyWindow("2")
    print("EXIT get_frame_face")


def get_camera_frame():
    global outputFrame1, lock
    while True:
        with lock:
            if outputFrame1 is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame1)
            if not flag:
                continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encodedImage.tobytes() + b'\r\n')


def get_camera_frame_face():
    global outputFrame2, lock
    while True:
        with lock:
            if outputFrame2 is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame2)
            if not flag:
                continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encodedImage.tobytes() + b'\r\n')


@router.get('/camera/raw', response_class=HTMLResponse)
async def camera_raw():
    """Маршрут потокового видео. Поместите это в атрибут src тега img."""
    return StreamingResponse(get_camera_frame(),
                             media_type='multipart/x-mixed-replace; boundary=frame')


@router.get('/camera/faces', response_class=HTMLResponse)
async def camera_faces():
    """Маршрут потокового видео. Поместите это в атрибут src тега img."""
    return StreamingResponse(get_camera_frame_face(),
                             media_type='multipart/x-mixed-replace; boundary=frame')
