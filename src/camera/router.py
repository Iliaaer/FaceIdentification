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


def get_frame():
    """Функция генератора потокового видео."""
    global outputFrame1, lock
    global cap
    print(0)
    while True:
        if not cap.isOpened():
            break
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("1", frame)
        shape1 = frame.shape
        key = cv2.waitKey(1) & 0xff
        with lock:
            outputFrame1 = frame.copy()
    cv2.destroyWindow("1")
    print("EXIT get_frame")


def get_frame_face():
    global outputFrame1, lock
    global outputFrame2
    global net, cap
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

        image_d = frame.copy()
        dets = net.detect(frame)

        res = []
        for i, b in enumerate(dets):
            if b[4] < 0.7:
                continue
            b_new = list(map(int, b))
            img_detect = image_d[b_new[1]:b_new[3], b_new[0]:b_new[2]]
            if 0 in img_detect.shape:
                continue
            res.append(b)

        image_arrays = []

        for i, b in enumerate(res):
            text = "{:.4f}".format(b[4])
            b_new = list(map(int, b))
            cv2.rectangle(frame, (b_new[0], b_new[1]), (b_new[2], b_new[3]), (0, 255, 0), 2)

        cv2.imshow("2", frame)
        key = cv2.waitKey(1) & 0xff

        with lock:
            outputFrame2 = frame.copy()

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

