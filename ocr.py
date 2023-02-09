from __future__ import print_function
# from imutils.video import WebcamVideoStream
# from imutils.video import FPS
import argparse
from threading import Thread


import cv2
import pytesseract as tr


tr.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# img = cv2.imread('img.png')

# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# print(tr.image_to_string(img))
# cv2.imshow("OCR",img)

# cv2.waitKey(0)

class WebCamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame 
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to inidicate if the thread 
        # should be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

# cap = WebcamVideoStream(src=0)
cap = WebCamVideoStream(src=0).start()

t = " "

while True:
    frame = cap.read()
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    print(type(tr.image_to_string(img)))
    print(tr.image_to_string(img))
    boxes = tr.image_to_boxes(img)
    t += tr.image_to_string(img) # also include any config options you use
    cv2.putText(
    img = frame,
    text = t,
    org = (50, 50),
    fontFace = cv2.FONT_HERSHEY_DUPLEX,
    fontScale = 0.5,
    color = (125, 246, 55),
    thickness = 1
    )
    # draw the bounding boxes on the image
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(frame, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)


    cv2.imshow("OCR",frame)

    k = cv2.waitKey(1)
    if k==27:
        cv2.destroyAllWindows()
        cap.stop()
        break
