from fer import FER
import cv2

img = cv2.imread("./images/train/happy/Training_87867.jpg")
detector = FER(mtcnn=True)
# print(detector.detect_emotions(img))

