import cv2

image = cv2.imread("Face/Olga.png")

image2 = image.copy()

mean, std = image2.mean(), image2.std()
img = (image2 - mean) / std

cv2.imshow("1", image)
cv2.imshow("2", img)

while cv2.waitKey(1):
    pass
# while cv2.waitKey(1) & str(0xff):
#     pass