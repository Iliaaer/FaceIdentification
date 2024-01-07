from datetime import datetime
import cv2

import verifications as vf
from RetinaFace.mobile_net import Net
from database.fill import get_user, get_userID_to_photo

db_reboot = False
db_path = "Face"
model_name = vf.VERIF.FACENET  # WORK
# model_name = vf.VERIF.FACENET512  # WORK
# model_name = vf.VERIF.SFACE
# model_name = vf.VERIF.ARCFACE
# model_name = vf.VERIF.DEEPFACE
# model_name = vf.VERIF.VGGFACE  # WORK

mode = vf.FaceVerification(model_name=model_name, db_path=db_path, db_reboot=db_reboot)

# modes = {
#     vf.VERIF.FACENET:    vf.FaceVerification(model_name=vf.VERIF.FACENET, db_path=db_path, db_reboot=db_reboot),
#     vf.VERIF.FACENET512: vf.FaceVerification(model_name=vf.VERIF.FACENET512, db_path=db_path, db_reboot=db_reboot),
#     vf.VERIF.SFACE:      vf.FaceVerification(model_name=vf.VERIF.SFACE, db_path=db_path, db_reboot=db_reboot),
#     vf.VERIF.ARCFACE:    vf.FaceVerification(model_name=vf.VERIF.ARCFACE, db_path=db_path, db_reboot=db_reboot),
#     vf.VERIF.DEEPFACE:   vf.FaceVerification(model_name=vf.VERIF.DEEPFACE, db_path=db_path, db_reboot=db_reboot),
#     vf.VERIF.VGGFACE:    vf.FaceVerification(model_name=vf.VERIF.VGGFACE, db_path=db_path, db_reboot=db_reboot),
# }
#
# mode = modes[model_name]

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    tm = cv2.TickMeter()
    net = Net(load2cpu=False)

    _is_recognition: bool = True

    while True:
        ret, frame = cap.read()
        tm.start()
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

            img_detect = image_d[b_new[1]:b_new[3], b_new[0]:b_new[2]]

            if _is_recognition:
                detect = mode.find(image=img_detect, distance_metric=vf.MT.COSINE)
                print(detect)

                if len(detect):
                    candidant = detect.iloc[0]
                    dst = detect.columns[-1]

                    path = candidant["identity"].replace("\\", "/")

                    user_id = get_userID_to_photo(path=path)
                    if user_id:
                        user = get_user(user_id)
                        if user:
                            full_name = user.full_name

                            cv2.putText(frame, str(user_id), (b_new[0], b_new[1] + 30), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 2)

            filename = f"Face/{datetime.now().strftime('%d%m%Y_%H%M%S')}_{str(i)}__{str(b[4])}.png"

            image_arrays.append([img_detect, filename, b[4]])

            cv2.rectangle(frame, (b_new[0], b_new[1]), (b_new[2], b_new[3]), (0, 255, 0), 2)

        tm.stop()
        cv2.putText(frame, 'FPS: {:.2f}'.format(tm.getFPS()), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255))

        cv2.imshow('fourcc', frame)
        tm.reset()

        key = cv2.waitKey(1) & 0xff

        if key == ord('s'):
            for img_detect, filename, _ in image_arrays:
                cv2.imwrite(filename, img_detect)
                print(f"[INFO] Save image to {filename}")

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
