import cv2
import mediapipe as mp
import time

class FaceDetector:
    def __init__(self, conf= 0.5, model=0):
        self.results = None
        self.conf = conf
        self.model = model

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection()


    def findFace(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):

                if draw:
                    rbbox = detection.location_data.relative_bounding_box
                    h, w, c = img.shape
                    cv2.rectangle(img, (int(rbbox.xmin * w), int(rbbox.ymin * h)),
                                  (int((rbbox.xmin + rbbox.width) * w), int((rbbox.ymin + rbbox.height) * h)), (210, 210, 0))
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (int(rbbox.xmin * w), int(rbbox.ymin * h - 20)),
                                cv2.FONT_HERSHEY_PLAIN, 3, (210, 210, 0), 2)
        return img


def main():
    cap = cv2.VideoCapture('./videos/2.mp4')

    ptime = 0
    ctime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img = detector.findFace(img)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()