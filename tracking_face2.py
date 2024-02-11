import cv2

from RetinaFace.mobile_net import Net
import verifications as vf
from database.fill import get_userID_to_photo, get_user
from tracker import VerificationTracker

THRESHOLD_FACE_DETECT = 0.7

model_name = vf.VERIF.DLIB
distance_metric = vf.MT.COSINE
mode = vf.FaceVerification(model_name=model_name)
mode.init_db(db_path="Face")

window_representations = 50

tracker = VerificationTracker(window_representations=window_representations)

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
            # print(len(rects))
            for rect in rects.items():
                object_id = rect[0]
                x1, y1, x2, y2 = rect[1]

                if [x1, y1, x2, y2] not in faces_detections:
                    continue

                c_x = int((x1 + x2) / 2)
                c_y = int((y1 + y2) / 2)

                text = f"ID {object_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.circle(frame, (c_x, c_y), 4, (0, 255, 0), -1)
                cv2.putText(frame, text, (c_x - 10, c_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        representations_face = tracker.get_representations_list()
        for i, representations in representations_face.items():
            if tracker.get_name(object_id=i):
                print(tracker.get_name(object_id=i))
                continue
            if len(representations) < window_representations:
                continue

            results = []
            # results = {}
            for representation in representations:
                detect = mode.find_representation(
                    target_representation=representation,
                    distance_metric=distance_metric
                )
                if not len(detect):
                    continue

                candidate = detect.iloc[0]
                path = "/".join(candidate["identity"].replace("\\", "/").split("/")[:-1])

                #     result_i = results.get(path, None)
                #     if not result_i:
                #         results[path] = {
                #             "Count": 1,
                #             "avg": candidate[f"{model_name}_{distance_metric.name}"],
                #         }
                #         continue
                #     results[path]["Count"] += 1
                #     results[path]["avg"] += candidate[f"{model_name}_{distance_metric.name}"]
                # for path in results.keys():
                #     results[path]["avg"] /= results[path]["Count"]
                # print(results)
                results.append(path)

            path = max(set(results), key=lambda x: results.count(x))
            user_id = get_userID_to_photo(path=path)
            user = get_user(user_id)
            print(path, results.count(path))
            print(results)
            full_name = user.full_name

            tracker.set_name(name=full_name, object_id=i)

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
    cv2.destroyAllWindows()
    cap.release()
