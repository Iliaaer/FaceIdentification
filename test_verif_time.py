import os
import time
import cv2

import verifications as vf


models_names = [vf.VERIF.FACENET, vf.VERIF.FACENET512, vf.VERIF.SFACE,
                vf.VERIF.ARCFACE, vf.VERIF.DEEPFACE, vf.VERIF.VGGFACE,
                vf.VERIF.DLIB]

model_name = models_names[6]

db_ = f"testFace/time100"

images = []
for r, _, f in os.walk(db_):
    for file in f:
        if ".jpg" in file.lower() or ".jpeg" in file.lower() or ".png" in file.lower():
            exact_path = r + "/" + file
            images.append(exact_path)
print(len(images))

mode = vf.FaceVerification(model_name=model_name)

image = cv2.imread(images[0])
detect = mode.represent_one(image=image)

print("[INFO] Start program")
start_time = time.time()
for _ in range(10):
    for name in images:
        image = cv2.imread(name)
        detect = mode.represent_one(image=image)

end_time = time.time() - start_time

print(f"[{model_name.name}] time = {str(end_time).replace('.', ',')}")
