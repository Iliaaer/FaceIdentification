import os
import time

import cv2
from RetinaFace.mobile_net import Net

net = Net(load2cpu=False)

pathDataset = r"testFace/dataset/201-291/20129109"

path_images = os.walk(pathDataset)

for r, _, f in path_images:
    for file in f:
        if file.lower().split(".")[-1] in ["jpg", "jpeg", "png"]:
            exact_path = r + "/" + file
            image = cv2.imread(exact_path)
            image_old = image.copy()
            dets = net.detect(image)

            res = []
            for i, b in enumerate(dets):
                if b[4] < 0.7:
                    continue
                b_new = list(map(int, b))
                img_detect = image_old[b_new[1]:b_new[3], b_new[0]:b_new[2]]
                if 0 in img_detect.shape:
                    continue
                res.append(b)
                cv2.imshow("1", img_detect)
                cv2.waitKey(1)
                time.sleep(5)

            print(exact_path, len(res))
            # b_new = list(map(int, dets))
            # img_detect = image_old[b_new[1]:b_new[3], b_new[0]:b_new[2]]
            # os.remove(exact_path)
            # cv2.imwrite(exact_path, img_detect)
