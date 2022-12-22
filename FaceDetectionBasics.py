import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('./videos/3.mp4')
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

ptime = 0
ctime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(id, detection)
            print(detection.score[0])
            # mpDraw.draw_detection(img, detection)

            rbbox = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            cv2.rectangle(img, (int(rbbox.xmin * w), int(rbbox.ymin * h)),
                          (int((rbbox.xmin + rbbox.width) * w), int((rbbox.ymin + rbbox.height) * h)), (210, 210, 0))
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (int(rbbox.xmin * w), int(rbbox.ymin * h - 20)),
                        cv2.FONT_HERSHEY_PLAIN, 3, (210, 210, 0), 2)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
