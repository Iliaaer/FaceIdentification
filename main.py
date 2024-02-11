from datetime import datetime
import cv2

import verifications as vf
from RetinaFace.mobile_net import Net
from database.fill import get_user, get_userID_to_photo

THRESHOLD_FACE_DETECT = 0.7

db_reboot = False
db_path = "Face"
models_names = [vf.VERIF.FACENET, vf.VERIF.FACENET512, vf.VERIF.SFACE,
                vf.VERIF.ARCFACE, vf.VERIF.DEEPFACE, vf.VERIF.VGGFACE,
                vf.VERIF.DLIB]
# model_name = vf.VERIF.FACENET  # WORK
# model_name = vf.VERIF.FACENET512  # WORK
# model_name = vf.VERIF.SFACE
# model_name = vf.VERIF.ARCFACE
# model_name = vf.VERIF.DEEPFACE
# model_name = vf.VERIF.VGGFACE  # WORK

# mode = vf.FaceVerification(model_name=model_name, db_path=db_path, db_reboot=db_reboot)

modes = {
    # vf.VERIF.FACENET: vf.FaceVerification(model_name=vf.VERIF.FACENET, db_path=db_path, db_reboot=db_reboot),
    # vf.VERIF.FACENET512: vf.FaceVerification(model_name=vf.VERIF.FACENET512, db_path=db_path, db_reboot=db_reboot),
    # vf.VERIF.SFACE: vf.FaceVerification(model_name=vf.VERIF.SFACE, db_path=db_path, db_reboot=db_reboot),
    # vf.VERIF.ARCFACE: vf.FaceVerification(model_name=vf.VERIF.ARCFACE, db_path=db_path, db_reboot=db_reboot),
    # vf.VERIF.DEEPFACE: vf.FaceVerification(model_name=vf.VERIF.DEEPFACE, db_path=db_path, db_reboot=db_reboot),
    # vf.VERIF.VGGFACE: vf.FaceVerification(model_name=vf.VERIF.VGGFACE, db_path=db_path, db_reboot=db_reboot),
    vf.VERIF.DLIB: vf.FaceVerification(model_name=vf.VERIF.DLIB, db_path=db_path, db_reboot=db_reboot),
}
#
mode = modes[models_names[6]]

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    tm = cv2.TickMeter()
    net = Net(load2cpu=False)

    ret, frame = cap.read()
    H, W = frame.shape[:2]

    _is_recognition: bool = True

    while True:
        ret, frame = cap.read()
        tm.start()
        image_d = frame.copy()
        detect_faces = net.detect(frame)

        image_arrays = []

        for i, detect_face in enumerate(detect_faces):
            if detect_face[4] < THRESHOLD_FACE_DETECT:
                continue
            detect_face = list(map(int, detect_face))
            x1, y1, x2, y2 = detect_face[:4]

            x1 = max(0, min(W, x1))
            x2 = max(0, min(W, x2))

            y1 = max(0, min(H, y1))
            y2 = max(0, min(H, y2))

            img_detect = frame[y1:y2, x1:x2]

            if _is_recognition:
                detect = mode.find_image(image=img_detect, distance_metric=vf.MT.COSINE)
                print(detect)

                if len(detect):
                    candidate = detect.iloc[0]
                    dst = detect.columns[-1]

                    path = "/".join(candidate["identity"].replace("\\", "/").split("/")[:-1])

                    print(path)
                    user_id = get_userID_to_photo(path=path)
                    if user_id:
                        user = get_user(user_id)
                        if user:
                            full_name = user.full_name

                            cv2.putText(frame, str(user_id), (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 2)

            filename = f"Face/{datetime.now().strftime('%d%m%Y_%H%M%S')}_{str(i)}.png"

            image_arrays.append([img_detect, filename])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        tm.stop()
        cv2.putText(frame, 'FPS: {:.2f}'.format(tm.getFPS()), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255))

        if _is_recognition:
            cv2.putText(frame, mode.model_name.name, (100, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        cv2.imshow('fourcc', frame)
        tm.reset()

        key = cv2.waitKey(1) & 0xff

        if key == ord('s'):
            for img_detect, filename, _ in image_arrays:
                cv2.imwrite(filename, img_detect)
                print(f"[INFO] Save image to {filename}")

        if key == ord("0"):
            _is_recognition = False
            print("[INFO] Recognition is dont work")

        for number_i in range(1, 7):
            if key == ord(str(number_i)):
                _is_recognition = True
                mode = modes[models_names[number_i - 1]]
                print(f"[START] model {models_names[number_i - 1].name}")

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
