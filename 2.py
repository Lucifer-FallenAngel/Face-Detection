import cv2
import mediapipe as mp
import time

# Create a named window and resize it
cv2.namedWindow("IMAGE", cv2.WINDOW_NORMAL)
cv2.resizeWindow("IMAGE", 1200, 650)

# Initialize video capture
cap = cv2.VideoCapture("4.mp4")

# Initialize MediaPipe Face Detection
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.5)

pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break  # Exit the loop if the video ends or cannot be read

    # Resize the image
    img = cv2.resize(img, (1280, 720))

    # Convert the image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image and detect faces
    results = faceDetection.process(imgRGB)

    # Draw face detection annotations on the image
    if results.detections:
        for id,detection in enumerate(results.detections):
            mpDraw.draw_detection(img, detection)
            #print(id, detection)
            #print(detection.score)
            #print(detection.location_data.relative_bounding_box)



    # Calculate and display the FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # Display the image
    cv2.imshow("IMAGE", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop if 'q' is pressed

# Release resources
cap.release()
cv2.destroyAllWindows()
