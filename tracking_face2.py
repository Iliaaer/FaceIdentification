import cv2

from RetinaFace.mobile_net import Net
import verifications as vf
from tracker import VerificationTracker


THRESHOLD_FACE_DETECT = 0.7

mode = vf.FaceVerification(model_name=vf.VERIF.SFACE)

tracker = VerificationTracker()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    tm = cv2.TickMeter()
    net = Net(load2cpu=False)

    ret, frame = cap.read()
    H, W = frame.shape[:2]

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        tm.start()
        detect_faces = net.detect(frame)

        faces_detections = []
        faces_representations = []
        for b in detect_faces:
            if b[4] < THRESHOLD_FACE_DETECT:
                continue
            b_new = list(map(int, b))
            x1, y1, x2, y2 = b_new[:4]

            x1 = max(0, min(W, x1))
            x2 = max(0, min(W, x2))

            y1 = max(0, min(H, y1))
            y2 = max(0, min(H, y2))

            img_detect = frame[y1:y2, x1:x2]

            detect = mode.represent_one(image=img_detect)

            faces_detections.append([x1, y1, x2, y2])
            faces_representations.append(detect)

        boxes_ids = tracker.update(faces_detections, faces_representations)

        if boxes_ids:
            rects, _ = boxes_ids
            for rect in rects.items():
                object_id = rect[0]
                x1, y1, x2, y2 = rect[1]
                c_x = int((x1 + x2) / 2)
                c_y = int((y1 + y2) / 2)

                text = f"ID {object_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.circle(frame, (c_x, c_y), 4, (0, 255, 0), -1)
                cv2.putText(frame, text, (c_x - 10, c_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        tm.stop()
        cv2.putText(frame, 'FPS: {:.2f}'.format(tm.getFPS()), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255))

        cv2.imshow('fourcc', frame)
        tm.reset()

        key = cv2.waitKey(1) & 0xff

        if key == ord('q'):
            break

    del net
    del mode
    cap.release()
    cv2.destroyAllWindows()
