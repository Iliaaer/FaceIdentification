import pickle as pk
import verifications as vf

test_number = 3
people_number = 20
db_reboot = False


# class VerificationFase(Enum):
#     VGGFACE: int = 0  # 2622
#     FACENET: int = 1  # 128
#     FACENET512: int = 2  # 512
#     DEEPFACE: int = 3  # 4096
#     ARCFACE: int = 4  # 512
#     SFACE: int = 5  # 128


model_name: vf.VERIF = vf.VERIF.SFACE

db_ = f"testFace/people{people_number}/test{test_number}"

db_path = f"{db_}/test"
db_valid_path = f"{db_}/validation"

file_name = f"representations_{model_name.name}.pkl".lower()

with open(f"{db_path}/{file_name}", "rb") as f:
    representations = pk.load(f)

i = representations[1]

print(len(i[1]))
