import pandas as pd
import json
import verifications as vf

test_number = 3
people_number = 20

db_ = f"testFace/people{people_number}/test{test_number}"

db_path = f"{db_}/test"
db_valid_path = f"{db_}/validation"

metrics = [vf.MT.COSINE, vf.MT.EUCLIDEAN, vf.MT.EUCLIDEAN_L2]

models_names = [vf.VERIF.FACENET, vf.VERIF.FACENET512, vf.VERIF.SFACE,
                vf.VERIF.ARCFACE, vf.VERIF.DEEPFACE, vf.VERIF.VGGFACE]

model_name = models_names[3]

metric = metrics[0]

with open(f"{db_}/json/result{test_number}_{model_name.name}_{metric.name}.json", 'r',
          encoding='utf-8') as f:
    data = json.load(f)

result = []
threshold = 0

while threshold <= 2:
    threshold += 0.005
    for i in range(len(data["Name"])):
        name, res = data["Name"][i], data["Result"][i]
        name1, value1 = res["Identity_name"], res["Value"]

        for _name1, _value1 in zip(name1, value1):
            c1 = name.split("\\")[1].split("/")[0] == _name1.split("'\'")[-1].split("\\")[1].split("/")[0]
            c2 = _value1 <= threshold

            result.append([name, _name1.split("'\'")[-1], _value1, _value1 <= threshold,
                           name.split("\\")[1].split("/")[0],
                           _name1.split("'\'")[-1].split("\\")[1].split("/")[0],
                           c1,
                           (int(c1) == 1 and int(c2) == 1)])

        result.append([None, None, None, None, None, None, None, None])

    df = pd.DataFrame(result, columns=["Name", "Identity", "Value", "Check", "Name1", "Name2", "Name1=Name2?", "Result"])

    df.to_excel(f"{db_}/xlsx/1/new_result{test_number}_{model_name.name}_{metric.name}_{str(threshold)}.xlsx")
