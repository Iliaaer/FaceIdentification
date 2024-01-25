import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

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

end_steps = 0

for i in data["Result"]:
    m = max(i["Value"])
    if m > end_steps:
        end_steps = m

print(end_steps)

results = []
threshold = 0
step = 0.005

pbar = tqdm(
    range(0, len(data["Name"])),
    desc="Finding representations"
)

for i in pbar:
    name, res = data["Name"][i], data["Result"][i]
    name1, value1 = res["Identity_name"], res["Value"]

    for _name1, _value1 in zip(name1, value1):
        c1 = name.split("\\")[1].split("/")[0] == _name1.split("'\'")[-1].split("\\")[1].split("/")[0]
        # result = [name, _name1.split("'\'")[-1], _value1]
        result = [_value1, name.split("\\")[1].split("/")[0], _name1.split("'\'")[-1].split("\\")[1].split("/")[0]]
        threshold = 0
        while threshold <= end_steps:
            c2 = float(_value1) <= float(threshold)

            result.append(c2 + c1 == 2 or (1-c2) + (1-c1) == 2)
            threshold += step
            threshold = round(threshold, 5)

        results.append(result)

    results.append([None] * len(results[-1]))

threshold = 0
a = ["Data", "Name1", "Name1"]
while threshold <= end_steps:
    a.append(threshold)
    threshold += step
    threshold = round(threshold, 5)

df = pd.DataFrame(results, columns=a)

x = a.copy()[3:]
print(x)
datas = [0] * len(x)

max_counter = 0
for i in results:
    if i[0] is None:
        continue
    for num in range(len(i)-3):
        datas[num] += int(i[3 + num])
    max_counter += 1

max_number = max(datas)
percentages = [number / max_number for number in datas]
print(datas)
print(percentages)
print(max_counter)

x = np.array(x)
percentages = np.array(percentages)
datas = np.array(datas)

plt.figure(figsize=(15, 5))
plt.plot(x, percentages, color='green', marker='o', markersize=2)
plt.title(model_name.name)
plt.grid(True)
plt.xticks(np.arange(0, end_steps, end_steps/20))
plt.xlim(0)
# plt.xlim(0, end_steps+0.2)

plt.savefig(f'{db_}/{model_name.name}.png', bbox_inches='tight', dpi=1200)
plt.clf()

# text = ""
# text += f"Сумма 1+2 МАКС: {max(check12)}\n"
# text += f"Соответствующий порог: {x[position]}\n"
# text += f"Соответствующий счёт 0: {check0[position]}\n"
# text += f"Соответствующий счёт 1: {check1[position]}\n"
# text += f"Относительное обнаружение: {check1[position] / check13[0]}\n"
# text += f"Отн. ошибка 0/(1+3): {check0[position] / check13[0]}\n"
# text += f"Позиция: {position}\n"
# text += f"Смещение: {mixing}"
#
# with open(f"{db_}/result/{model_name.name}/{model_name.name}.txt", "w") as file:
#     file.write(text)


# plt.figure(figsize=(15, 5))
# plt.plot(x, check1, label='Счёт 1')
# plt.plot(x, check2, label='Счёт 2')
# plt.plot(x, check12, label='Сумма 1+2')
# plt.plot(x, check3, label='Счёт 3')
# plt.plot(x, check0, label='Счёт 0')
# plt.title(model_name.name)
# plt.grid(True)
# plt.xticks(np.arange(0, end_steps, end_steps/20))
# plt.xlim(0)
# plt.legend()
# plt.savefig(f'{db_}/result/{model_name.name}/new_{model_name.name}1.png', bbox_inches='tight', dpi=1200)
# plt.clf()
#
# plt.figure(figsize=(15, 5))
# plt.plot(x, check1_percentages, label='Счёт 1%')
# plt.plot(x, check2_percentages, label='Счёт 2%')
# plt.plot(x, check12_percentages, label='Сумма 1+2%')
# plt.title(model_name.name)
# plt.grid(True)
# plt.xticks(np.arange(0, end_steps, end_steps/20))
# plt.xlim(0)
# plt.legend()
# plt.savefig(f'{db_}/result/{model_name.name}/new_{model_name.name}2.png', bbox_inches='tight', dpi=1200)
# plt.clf()
#
# plt.figure(figsize=(15, 5))
# plt.plot(x, check12, label='Сумма 1+2')
# plt.plot(x, check30, label='Сумма 3+0')
# plt.title(model_name.name)
# plt.grid(True)
# plt.xticks(np.arange(0, end_steps, end_steps/20))
# plt.xlim(0)
# plt.legend()
# plt.savefig(f'{db_}/result/{model_name.name}/new_{model_name.name}3.png', bbox_inches='tight', dpi=1200)
# plt.clf()
