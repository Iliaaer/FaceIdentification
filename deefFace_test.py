from os import listdir
from keras.preprocessing import image
import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, LocallyConnected2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.preprocessing.image import img_to_array

target_size = (152, 152)


def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def readImg(img_path):
    img = cv2.resize(cv2.imread(img_path), target_size)
    img_pixels = img_to_array(img)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255
    return img_pixels


if __name__ == "__main__":
    base_model = Sequential()
    base_model.add(Convolution2D(32, (11, 11), activation='relu', name='C1', input_shape=(152, 152, 3)))
    base_model.add(MaxPooling2D(pool_size=3, strides=2, padding='same', name='M2'))
    base_model.add(Convolution2D(16, (9, 9), activation='relu', name='C3'))
    base_model.add(LocallyConnected2D(16, (9, 9), activation='relu', name='L4'))
    base_model.add(LocallyConnected2D(16, (7, 7), strides=2, activation='relu', name='L5'))
    base_model.add(LocallyConnected2D(16, (5, 5), activation='relu', name='L6'))
    base_model.add(Flatten(name='F0'))
    base_model.add(Dense(4096, activation='relu', name='F7'))
    base_model.add(Dropout(rate=0.5, name='D0'))
    base_model.add(Dense(8631, activation='softmax', name='F8'))

    base_model.load_weights("model/VGGFace2_DeepFace_weights_val-0.9034.h5")
    model = Model(inputs=base_model.layers[0].input, outputs=base_model.layers[-3].output)

    # name = ["15112023_164148_0__0.9964678.png", "15112023_164206_0__0.9992095.png", "15112023_164208_0__0.9992194.png",
    #         "15112023_164209_0__0.9989268.png", "15112023_164210_0__0.9981596.png", "15112023_164211_0__0.9982539.png",
    #         "15112023_164212_0__0.9980386.png", "15112023_164213_0__0.99848735.png"]

    name = ["15112023_164213_0__0.99848735.png"]

    employees = {}
    for i in name:
        images = readImg(f"Face/{i}")
        representation = model.predict(images)[0]
        employees[i] = representation

    image_2 = readImg("Face/Zarubin_Ilya/15112023_164214_0__0.998481.png")
    captured_representation = model.predict(image_2)[0]

    # distance = findEuclideanDistance(l2_normalize(captured_representation), l2_normalize(im1))

    # print(distance)

    distances = {}
    for i in employees:
        source_representation = employees[i]

        distance = findEuclideanDistance(l2_normalize(captured_representation), l2_normalize(source_representation))
        distances[i] = distance

    print(distances)

    # employee_pictures = "database/"
    #
    # employees = dict()
    #
    # for file in listdir(employee_pictures):
    #     employee, extension = file.split(".")
    #     img_path = 'database/%s.jpg' % (employee)
    #     # img = detectFace(img_path)
    #
    #     representation = model.predict(img)[0]
    #
    #     employees[employee] = representation
