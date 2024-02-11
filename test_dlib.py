import json
import os

import cv2
from tqdm import tqdm

import verifications as vf

metrics = [vf.MT.COSINE, vf.MT.EUCLIDEAN, vf.MT.EUCLIDEAN_L2]

pbar_people = tqdm(
    [1, 2, 5, 10, 15, 20, 30],
    desc="People number go",
    colour="green"
)
pbar_number = tqdm(
    [1, 2, 3, 4],
    desc="Test number go",
    colour="red"
)

mode = vf.FaceVerification(model_name=vf.VERIF.DLIB)


tttt = []
for people_number in pbar_people:
    for test_number in pbar_number:
        db_ = f"testFace/people{people_number}/test{test_number}"

        dlib_json = f"dlib_test/people{people_number}/test{test_number}"

        if not os.path.isdir(f"dlib_test/people{people_number}"):
            os.mkdir(f"dlib_test/people{people_number}")
        if not os.path.isdir(f"dlib_test/people{people_number}/test{test_number}"):
            os.mkdir(f"dlib_test/people{people_number}/test{test_number}")

        db_path = f"{db_}/test"
        db_valid_path = f"{db_}/validation"

        images = []
        for r, _, f in os.walk(db_valid_path):
            for file in f:
                if ".jpg" in file.lower() or ".jpeg" in file.lower() or ".png" in file.lower():
                    exact_path = r + "/" + file
                    images.append(exact_path)
        print(len(images))

        if not os.path.isdir(f"{db_}/json"):
            os.mkdir(f"{db_}/json")

        mode.init_db(db_path=db_path)

        for metric in metrics:
            print(people_number, test_number, metric.name)

            result = []

            data = {
                "Name": [],
                "Result": []
            }
            image = cv2.imread(images[0])
            detect = mode.represent_one(image=image)

            print("[INFO START]")
            for name in images:
                image = cv2.imread(name)
                detect = mode.find_image(image=image, distance_metric=metric)

                data["Name"].append(name)
                res = {"Identity_name": [], "Value": []}
                for i in detect.values:
                    res["Identity_name"].append(i[0])
                    res["Value"].append(i[1])
                data["Result"].append(res)

            if not os.path.isdir(f"{dlib_json}/json"):
                os.mkdir(f"{dlib_json}/json")

            with open(f"{dlib_json}/json/result{test_number}_{vf.VERIF.DLIB.name}_{metric.name}.json", 'w',
                      encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            del data, result, res, image, detect
