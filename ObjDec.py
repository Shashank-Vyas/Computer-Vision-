from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


vid = cv2.VideoCapture('rtsp://admin:admin@192.168.29.37:1935')
vid.set(3, 640)
vid.set(4, 480)
model = YOLO("YOLO weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane",
                  "bus", "train", "truck", "boat", "traffic light", 
                  "fire hydrant", "stop sign", "parking meter", "bench", 
                  "bird", "cat", "dog", "horse", "sheep", "cow", 
                  "elephant", "bear", "zebra", "giraffe", "backpack", 
                  "umbrella", "handbag", "tie", "suitcase", "frisbee", 
                  "skis", "snowboard", "sports ball", "kite", 
                  "baseball bat", "baseball glove", "skateboard", 
                  "surfboard", "tennis racket", "bottle", "wine glass", 
                  "cup", "fork", "knife", "spoon", "bowl", "banana", 
                  "apple", "sandwich", "orange", "broccoli", "carrot", 
                  "hot dog", "pizza", "donut", "cake", "chair", "sofa", 
                  "potted plant", "bed", "dining table", "toilet", 
                  "tv monitor", "laptop", "mouse", "remote", "keyboard", 
                  "cell phone", "microwave", "oven", "toaster", "sink", 
                  "refrigerator", "book", "clock", "vase", "scissors", 
                  "teddy bear", "hair drier", "toothbrush"
                  ]

prevTime = 0
while vid.isOpened():
    success, frame = vid.read()
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2-x1, y2-y1
            cv2.rectangle(frame,(x1, y1, w, h), (255, 0, 0))
            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)
            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(frame, f'{classNames[cls]}' f'{conf}', (max(0, x1),max(35, y1)), scale=1, thickness=2)
            

    cv2.imshow("Image", frame)
    # cv2.imshow("ImageRegion", imgRegion)
    if cv2.waitKey(1) & 0XFF == ord('1'):
        break