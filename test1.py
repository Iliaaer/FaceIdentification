import gc
import json
import os
import time

import cv2
from tqdm import tqdm

import verifications as vf
from verifications.until import distance as dst

metrics = [vf.MT.COSINE, vf.MT.EUCLIDEAN, vf.MT.EUCLIDEAN_L2]
# metrics = [vf.MT.COSINE]

models_names = [vf.VERIF.FACENET, vf.VERIF.FACENET512, vf.VERIF.SFACE,
                vf.VERIF.ARCFACE, vf.VERIF.DEEPFACE, vf.VERIF.VGGFACE]

# models_names = [vf.VERIF.FACENET, vf.VERIF.FACENET512, vf.VERIF.SFACE]
# models_names = [vf.VERIF.ARCFACE, vf.VERIF.DEEPFACE, vf.VERIF.VGGFACE]
# models_names = [vf.VERIF.DEEPFACE]
# models_names = [vf.VERIF.DEEPFACE, vf.VERIF.VGGFACE]

pbar_people = tqdm(
    [100],
    desc="People number go",
    colour="green"
)
pbar_number = tqdm(
    [4],
    desc="Test number go",
    colour="red"
)
# test_number = 3
# people_number = 20
tttt = []
for people_number in pbar_people:
    for test_number in pbar_number:
        db_reboot = False

        db_ = f"testFace/people{people_number}/test{test_number}"

        db_path = f"{db_}/test"
        db_valid_path = f"{db_}/validation"

        # modes = {
        #     vf.VERIF.DEEPFACE: vf.FaceVerification(model_name=vf.VERIF.DEEPFACE, db_path=db_path, db_reboot=db_reboot),
        #     vf.VERIF.FACENET: vf.FaceVerification(model_name=vf.VERIF.FACENET, db_path=db_path, db_reboot=db_reboot),
        #     vf.VERIF.FACENET512: vf.FaceVerification(model_name=vf.VERIF.FACENET512, db_path=db_path,
        #                                              db_reboot=db_reboot),
        #     vf.VERIF.SFACE: vf.FaceVerification(model_name=vf.VERIF.SFACE, db_path=db_path, db_reboot=db_reboot),
        #     vf.VERIF.ARCFACE: vf.FaceVerification(model_name=vf.VERIF.ARCFACE, db_path=db_path, db_reboot=db_reboot),
        #     vf.VERIF.VGGFACE: vf.FaceVerification(model_name=vf.VERIF.VGGFACE, db_path=db_path, db_reboot=db_reboot),
        # }

        images = []
        for r, _, f in os.walk(db_valid_path):
            for file in f:
                if ".jpg" in file.lower() or ".jpeg" in file.lower() or ".png" in file.lower():
                    exact_path = r + "/" + file
                    images.append(exact_path)
        print(len(images))
        # print(*images)

        # if not os.path.isdir(f"{db_}/xlsx"):
        #     os.mkdir(f"{db_}/xlsx")

        if not os.path.isdir(f"{db_}/json"):
            os.mkdir(f"{db_}/json")

        pbar_models = tqdm(
            models_names,
            desc="MODELS names go",
            colour="green"
        )
        start_time = 0
        for model_name in pbar_models:

            # mode = modes[model_name]
            # mode = vf.FaceVerification(model_name=model_name, db_path=db_path, db_reboot=db_reboot)
            for metric in metrics:
                gc.collect()
                mode = vf.FaceVerification(model_name=model_name, db_path=db_path, db_reboot=db_reboot)
                print(people_number, test_number, model_name.name, metric.name)

                result = []

                data = {
                    "Name": [],
                    "Result": []
                }
                image = cv2.imread(images[0])
                detect = mode.find(image=image, distance_metric=metric)

                start_time = time.time()
                print("[INFO START]")
                for name in images:
                    image = cv2.imread(name)
                    detect = mode.find(image=image, distance_metric=metric)

                    data["Name"].append(name)
                    res = {"Identity_name": [], "Value": []}
                    for i in detect.values:
                        res["Identity_name"].append(i[0])
                        res["Value"].append(i[1])
                    data["Result"].append(res)

                end_time = time.time() - start_time
                tttt.append(f"[TIME] work program{people_number} and {test_number} ({model_name.name} and {metric.name}) is {str(end_time).replace('.', ',')}")
                print(f"[TIME] work program ({model_name.name} and {metric.name}) is {str(end_time).replace('.', ',')}")

                threshold = dst.findThreshold(model_name, metric)
                # print(data)

                result = []
                for i in range(len(images)):
                    name, res = data["Name"][i], data["Result"][i]
                    name1, value1 = res["Identity_name"], res["Value"]

                    for _name1, _value1 in zip(name1, value1):
                        result.append([name, _name1.split("'\'")[-1], _value1, _value1 <= threshold])

                    result.append([None, None, None, None])

                # df = pd.DataFrame(result, columns=["Name", "Identity", "Value", "Check"])

                # df.to_excel(f"{db_}/xlsx/result{test_number}_{model_name.name}_{metric.name}.xlsx")

                with open(f"{db_}/json/result{test_number}_{model_name.name}_{metric.name}.json", 'w',
                          encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

                # del mode, df, data, result, res
                del mode, data, result, res, image, detect

                gc.collect()


for t in tttt:
    print(t)


"""
import gc
import json
import os
import time

for people_number in [1, 2, 5, 10, 15, 20, 30]:
    for test_number in [1, 2, 3, 4]:


        db_ = f"testFace/people{people_number}/test{test_number}"

        db_path = f"{db_}/test"
        db_valid_path = f"{db_}/validation"

        

        images2 = []
        for r, _, f in os.walk(db_path):
            for file in f:
                if ".jpg" in file.lower() or ".jpeg" in file.lower() or ".png" in file.lower():
                    exact_path = r + "/" + file
                    images2.append(exact_path)


        images1 = []
        for r, _, f in os.walk(db_valid_path):
            for file in f:
                if ".jpg" in file.lower() or ".jpeg" in file.lower() or ".png" in file.lower():
                    exact_path = r + "/" + file
                    images1.append(exact_path)
        print(people_number, test_number, "test", len(images2), "valid", len(images1))
    print("\n")
        

"""
