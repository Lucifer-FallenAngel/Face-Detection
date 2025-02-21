import cv2
import mediapipe as mp
import time

cv2.namedWindow("IMAGE", cv2.WINDOW_NORMAL)
cv2.resizeWindow("IMAGE", 1200, 650)

cap = cv2.VideoCapture("2.mp4")

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

pTime = 0

while True:
    success, img = cap.read()

    img = cv2.resize(img, (1280, 720))

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    #print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            print(id, detection)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
    cv2.imshow("IMAGE", img)
    cv2.waitKey(1)