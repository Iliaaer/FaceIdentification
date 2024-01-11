import os
import cv2
import numpy as np
import pandas as pd
import json

from matplotlib import pyplot as plt

import verifications as vf
from verifications.until import distance as dst

metrics = [vf.MT.COSINE, vf.MT.EUCLIDEAN, vf.MT.EUCLIDEAN_L2]

models_names = [vf.VERIF.FACENET, vf.VERIF.FACENET512, vf.VERIF.SFACE,
                vf.VERIF.ARCFACE, vf.VERIF.DEEPFACE, vf.VERIF.VGGFACE]

# models_names = [vf.VERIF.FACENET, vf.VERIF.FACENET512, vf.VERIF.SFACE]
# models_names = [vf.VERIF.ARCFACE, vf.VERIF.DEEPFACE, vf.VERIF.VGGFACE]

test_number = 1
people_number = 1
db_reboot = False

db_ = f"testFace/people{people_number}/test{test_number}"

db_path = f"{db_}/test"
db_valid_path = f"{db_}/validation"

modes = {
    # vf.VERIF.DEEPFACE: vf.FaceVerification(model_name=vf.VERIF.DEEPFACE, db_path=db_path, db_reboot=db_reboot),
    vf.VERIF.FACENET: vf.FaceVerification(model_name=vf.VERIF.FACENET, db_path=db_path, db_reboot=db_reboot),
    vf.VERIF.FACENET512: vf.FaceVerification(model_name=vf.VERIF.FACENET512, db_path=db_path, db_reboot=db_reboot),
    vf.VERIF.SFACE: vf.FaceVerification(model_name=vf.VERIF.SFACE, db_path=db_path, db_reboot=db_reboot),
    vf.VERIF.ARCFACE: vf.FaceVerification(model_name=vf.VERIF.ARCFACE, db_path=db_path, db_reboot=db_reboot),
    vf.VERIF.VGGFACE: vf.FaceVerification(model_name=vf.VERIF.VGGFACE, db_path=db_path, db_reboot=db_reboot),
}

images = []
for r, _, f in os.walk(db_valid_path):
    for file in f:
        if ".jpg" in file.lower() or ".jpeg" in file.lower() or ".png" in file.lower():
            exact_path = r + "/" + file
            images.append(exact_path)
print(*images)

if not os.path.isdir(f"{db_}/xlsx"):
    os.mkdir(f"{db_}/xlsx")

if not os.path.isdir(f"{db_}/json"):
    os.mkdir(f"{db_}/json")

# for model_name in models_names[0]:
model_name = models_names[0]
mode = modes[model_name]
# mode = vf.FaceVerification(model_name=model_name, db_path=db_path, db_reboot=db_reboot)
# print(model_name.name, metric.name)
# model_name: vf.VERIF = models_names

result = []

data = {
    "Name": [],
    "Result": []
}

for name in images[:2]:
    image = cv2.imread(name)
    target_obj = vf.changed_face_size(img=image, target_size=mode.target_size)

    face = vf.get_normalize_image(target_obj, model_name)

    face = np.expand_dims(face, axis=0)
    result.append(mode.model.predict(face)[0])

distance_vector = np.square(result[0] - result[1])
distance = np.sqrt(distance_vector.sum())
print("Euclidean distance: ", distance)

img1_graph = []
img2_graph = []
distance_graph = []

for i in range(0, 200):
    img1_graph.append(result[0])
    img2_graph.append(result[1])
    distance_graph.append(distance_vector)

img1_graph = np.array(img1_graph)
img2_graph = np.array(img2_graph)
distance_graph = np.array(distance_graph)

fig = plt.figure()

ax1 = fig.add_subplot(3, 2, 1)
plt.imshow(cv2.imread(images[0]))
plt.axis("off")

ax2 = fig.add_subplot(3, 2, 2)
im = plt.imshow(img1_graph, interpolation="nearest", cmap=plt.cm.ocean)
plt.colorbar()

ax3 = fig.add_subplot(3, 2, 3)
plt.imshow(cv2.imread(images[1]))
plt.axis("off")

ax4 = fig.add_subplot(3, 2, 4)
im = plt.imshow(img2_graph, interpolation="nearest", cmap=plt.cm.ocean)
plt.colorbar()

ax5 = fig.add_subplot(3, 2, 5)
plt.text(0.35, 0, f"Distance: {distance}")
plt.axis("off")

ax6 = fig.add_subplot(3, 2, 6)
im = plt.imshow(distance_graph, interpolation="nearest", cmap=plt.cm.ocean)
plt.colorbar()

plt.show()

    # print(len(result))
    # exit()

#     detect = mode.find(image=image, distance_metric=metric)
#
#     data["Name"].append(name)
#     res = {"Identity_name": [], "Value": []}
#     for i in detect.values:
#         res["Identity_name"].append(i[0])
#         res["Value"].append(i[1])
#     data["Result"].append(res)
#
# threshold = dst.findThreshold(model_name, metric)
# # print(data)
#
# result = []
# for i in range(len(images)):
#     name, res = data["Name"][i], data["Result"][i]
#     name1, value1 = res["Identity_name"], res["Value"]
#
#     for _name1, _value1 in zip(name1, value1):
#         result.append([name, _name1.split("'\'")[-1], _value1, _value1 <= threshold])
#
#     result.append([None, None, None, None])
#
# df = pd.DataFrame(result, columns=["Name", "Identity", "Value", "Check"])
#
# df.to_excel(f"{db_}/xlsx/result{test_number}_{model_name.name}_{metric.name}.xlsx")
#
# with open(f"{db_}/json/result{test_number}_{model_name.name}_{metric.name}.json", 'w',
#           encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)
