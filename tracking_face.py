import cv2

from tracker import CentroidTracker
from RetinaFace.mobile_net import Net

tracker = CentroidTracker()

THRESHOLD_FACE_DETECT = 0.7

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    tm = cv2.TickMeter()
    net = Net(load2cpu=False)

    while True:
        ret, frame = cap.read()
        tm.start()
        image_d = frame.copy()
        detect_faces = net.detect(frame)

        res = []
        for i, b in enumerate(detect_faces):
            if b[4] < THRESHOLD_FACE_DETECT:
                continue
            b_new = list(map(int, b))
            img_detect = image_d[b_new[1]:b_new[3], b_new[0]:b_new[2]]
            if 0 in img_detect.shape:
                continue
            res.append(b)

        faces_detections = []
        for i, b in enumerate(res):
            text = "{:.4f}".format(b[4])
            b_new = list(map(int, b))
            x1, y1, x2, y2 = b_new[:4]
            faces_detections.append([x1, y1, x2, y2])
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        boxes_ids = tracker.update(faces_detections)
        print(len(faces_detections), end="--")
        if boxes_ids:
            print(len(boxes_ids[0]))
            objects, rects = boxes_ids
            # print(len(objects))
            for object, rect in zip(objects.items(), rects.items()):
                objectID, centroid = object
                x1, y1, x2, y2 = rect[1]
                text = f"ID {objectID}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        tm.stop()
        cv2.putText(frame, 'FPS: {:.2f}'.format(tm.getFPS()), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255))

        cv2.imshow('fourcc', frame)
        tm.reset()

        key = cv2.waitKey(1) & 0xff

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
