import cv2 as cv

cap = cv.VideoCapture(0)

while cap.isOpened():

    success, image = cap.read()

    if not success:
        break

    cv.imshow("frame", image)
    if cv.waitKey(5) & 0xFF == 27:
        break

cap.release()