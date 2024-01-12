import cv2
cam = cv2.VideoCapture(4)
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed")
        break
    cv2.imshow('test',frame)
    cv2.waitKey(100)
    