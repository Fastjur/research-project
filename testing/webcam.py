import cv2
import time
from fer import FER

cap = cv2.VideoCapture(0)
detector = FER(mtcnn=True)

t_end = time.time() + 20
while time.time() < t_end:
    ret, frame = cap.read()
    faces = detector.detect_emotions(frame)

    for i in range(len(faces)):
        box = faces[i]['box']
        x = box[0]
        y = box[1]
        width = box[2]
        height = box[3]

        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

        offset = 32
        for emotion in faces[i]['emotions']:
            text = f"{emotion}: {faces[i]['emotions'][emotion]}"
            cv2.putText(frame, text, (x, y + height + offset), 0, 1, (0, 0, 255), 2)
            offset += 32

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
