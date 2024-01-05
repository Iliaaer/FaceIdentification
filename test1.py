import os
import cv2
import verifications as vf

db_reboot = False
db_path = "testFace/test_1"
db_valid_path = "testFace/validation_1/20129101"
model_name = vf.VERIF.VGGFACE  # WORK

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

images = os.listdir(db_valid_path)

for name in images:
    image = cv2.imread(db_valid_path + "/" + name)
    detect = mode.find(image=image, distance_metric=vf.MT.COSINE)
    val = None
    if detect.values.any():
        val = detect.values[0][1]
    print(name, len(detect.values), val)

# print(images)
