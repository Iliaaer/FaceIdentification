import cv2
import numpy as np
import os
import pickle as pk

import pandas as pd
from tqdm import tqdm

from verifications.basemodels import (
    VGGFace,
    Facenet,
    Facenet512,
    DeepFace,
    ArcFace,
    SFace,
)
from verifications.until.transfers import Metric as MT
from verifications.until.transfers import VerificationFase as VERIF
from verifications.until.functions import changed_face_size, get_target_size, init_folder, get_normalize_image
from verifications.until import distance as dst

pd.options.display.float_format = '{:.5f}'.format


class FaceVerification:
    def __init__(self, model_name: VERIF, db_path: str, db_reboot: bool = False):
        models = {
            VERIF.VGGFACE: VGGFace.loadModel,
            VERIF.FACENET: Facenet.loadModel,
            VERIF.FACENET512: Facenet512.loadModel,
            VERIF.DEEPFACE: DeepFace.loadModel,
            VERIF.ARCFACE: ArcFace.loadModel,
            VERIF.SFACE: SFace.load_model,

            # "OpenFace": OpenFace.loadModel,
            # "DeepID": DeepID.loadModel,
            # "Dlib": DlibWrapper.loadModel,
        }

        self.representations = []

        self.model_name = model_name.name

        print(model_name.name)

        self.model = models.get(model_name)

        if not self.model:
            raise ValueError(f"Invalid model_name passed - {model_name}")

        self.target_size = get_target_size(model_name)

        self.model = self.model()

        self.initDB(db_path=db_path, db_reboot=db_reboot)

    def __represent(self, image: np.ndarray) -> pd.DataFrame:
        """
        This function represents facial image as vector. The function uses convolutional neural
        networks models to generate vector embeddings.

        Parameters:
                image (np.ndarray): numpy array (BGR)
                encoded images could be passed. Source image can have many faces. Then, result will
                be the size of number of faces appearing in the source image.

        Returns:
                Represent function returns a list of object with multidimensional vector (embedding).
                The number of dimensions is changing based on the reference model.
                E.g. FaceNet returns 128 dimensional vector; VGG-Face returns 2622 dimensional vector.
        """

        face = image.copy()

        # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # if self.model_name == VERIF.FACENET.name:
        #     face = face.astype('float32')
        #     face = (face - face.mean()) / face.std()
        #
        # if self.model_name == VERIF.ARCFACE:
        #     face = face.astype('float32')
        #     # face /= 127.5
        #     # face -= 1
        #     face -= 127.5
        #     face /= 128

        # if self.model_name == VERIF.ARCFACE:
        #     face -= 127.5
        #     face /= 128

        face = get_normalize_image(face, self.model_name)

        face = np.expand_dims(face, axis=0)

        return self.model.predict(face)[0]

        # img = changed_face_size(img, target_size=self.target_size, grayscale=False)
        # # img = cv2.resize(img, self.target_size)
        #
        # img = np.expand_dims(img, axis=0)
        # if img.max() > 1:
        #     img /= 255
        #
        # return self.model.predict(img)[0].tolist()

    def initDB(self, db_path: str, db_reboot: bool = False):
        if not os.path.isdir(db_path):
            raise ValueError("Passed db_path does not exist!")

        file_name = f"representations_{self.model_name}.pkl".lower()

        if db_reboot and os.path.exists(f"{db_path}/{file_name}"):
            os.remove(f"{db_path}/{file_name}")

        employees = []
        employees_not = []

        for r, _, f in os.walk(db_path):
            for file in f:
                if ".jpg" in file.lower() or ".jpeg" in file.lower() or ".png" in file.lower():
                    exact_path = r + "/" + file
                    employees.append(exact_path)

        if len(employees) == 0:
            raise ValueError(f"There is no image in {db_path} folder! Validate .jpg or .png files exist in this "
                             f"path.")

        if os.path.exists(f"{db_path}/{file_name}"):
            with open(f"{db_path}/{file_name}", "rb") as f:
                self.representations = pk.load(f)

            representations_name = [pf[0] for pf in self.representations]
            for path_file in employees:
                if path_file not in representations_name:
                    # employees_not.append(path_file)
                    self.representations = []
                    os.remove(f"{db_path}/{file_name}")
                    break

        if not os.path.exists(f"{db_path}/{file_name}"):

            pbar = tqdm(
                range(0, len(employees)),
                desc="Finding representations"
            )

            for index in pbar:
                employee = employees[index]

                img_content = changed_face_size(img=cv2.imread(employee),
                                                target_size=self.target_size)
                img_representation = self.__represent(image=img_content)

                self.representations.append([employee, img_representation])

            with open(f"{db_path}/{file_name}", "wb") as f:
                pk.dump(self.representations, f)

        self.df = pd.DataFrame(self.representations, columns=["identity", f"{self.model_name}_representation"])

    def find(self, image: np.ndarray, distance_metric: MT = MT.COSINE):
        target_obj = changed_face_size(img=image,
                                       target_size=self.target_size)

        target_representation = self.__represent(image=target_obj)

        result_df = self.df.copy()

        distances = []
        for index, instance in result_df.iterrows():
            source_representation = instance[f"{self.model_name}_representation"]

            if distance_metric == MT.COSINE:
                distance = dst.findCosineDistance(source_representation, target_representation)
            elif distance_metric == MT.EUCLIDEAN:
                distance = dst.findEuclideanDistance(source_representation, target_representation)
            elif distance_metric == MT.EUCLIDEAN_L2:
                distance = dst.findEuclideanDistance(
                    dst.l2_normalize(source_representation),
                    dst.l2_normalize(target_representation),
                )
            else:
                raise ValueError(f"invalid distance metric passes - {distance_metric}")

            # distance = np.linalg.norm(source_representation - target_representation)

            distances.append(distance)

        result_df[f"{self.model_name}_{distance_metric.name}"] = distances
        threshold = dst.findThreshold(self.model_name, distance_metric)
        result_df = result_df.drop(columns=[f"{self.model_name}_representation"])
        result_df = result_df[result_df[f"{self.model_name}_{distance_metric.name}"] <= threshold]  # search res dst_min

        # result_df = result_df[result_df[f"{self.model_name}_{distance_metric.name}"] <= 8]  # or take linalg.norm

        result_df = result_df.sort_values(
            by=[f"{self.model_name}_{distance_metric.name}"], ascending=True
        ).reset_index(drop=True)

        return result_df


init_folder()

"""

verify
find

"""
