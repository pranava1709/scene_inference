import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
from pixellib.semantic import semantic_segmentation
import numpy as np
import cv2
import pytesseract as tr


ocrWIN = np.ones([50,500])

tr.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

cap = cv2.VideoCapture(1)

ins = instanceSegmentation()
ins.load_model("pointrend_resnet50.pkl",detection_speed = "rapid")

#t =  r'C:\Users\PRANAVA SETH\Anaconda3\Lib\site-packages\Tesseract-OCR\tesseract.exe'
#t = r' C:\Program Files\Tesseract-OCR\tesseract.exe'
t = ''
while True:
    _,frame = cap.read()
    cv2.imwrite("image.jpg",frame)
    ins.segmentImage("image.jpg", show_bboxes=True, output_image_name="output_image.jpg")
    img = cv2.imread("output_image.jpg")
    imgG = frame.copy()
    imgG = cv2.cvtColor(imgG,cv2.COLOR_BGR2GRAY)

    imgOCR = frame.copy()
    imgOCR = cv2.cvtColor(imgOCR,cv2.COLOR_BGR2RGB)
    # t = " "
    t = tr.image_to_string(imgOCR)

    edges = cv2.Canny(imgG,100,200)

    cv2.putText(
    img = ocrWIN,
    text = t,
    org = (50, 15),
    fontFace = cv2.FONT_HERSHEY_DUPLEX,
    fontScale = 0.6,
    color = (0, 0, 0),
    thickness = 1
    )

    cv2.imshow("Segmentation",img)
    cv2.imshow("Boundary",edges)
    cv2.imshow("OCR",ocrWIN)
    ocrWIN = np.ones([50,500])
    k = cv2.waitKey(1)
    if k==27:
        cap.release()
        break