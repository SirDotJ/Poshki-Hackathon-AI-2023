import cv2
import numpy
import numpy as np
import os
from pathlib import Path

CONFIDENCE = 0.3
NMS_THRESHOLD = 0.4
SCORE_THRESHOLD = 0.5
MINIMUM_WIDTH = 20
MINIMUM_HEIGHT = 40

whT = 320

# конфигурация нейронной сети
config_path = "./model/config/yolov3.cfg"
# файл весов сети YOLO
weights_path = "./model/weights/yolov3.weights"
# weights_path = "weights/yolov3-tiny.weights"
video_path = "./test/input/11_50_20.mp4"

output_frames_path = "./test/output/frames/"
output_video_path = "./test/output/output.mp4"



def findObjects(outputs, image):
    hT, wT, cT = image.shape
    boundingBox = []
    classIds = []
    confidences = []

    # через все три output
    for output in outputs:
        # через каждую область обнаружения
        for detection in output:
            scores = detection[5:]
            classId = numpy.argmax(scores)
            if(classId != 0):
                continue
            confidence = scores[classId]
            if (confidence > CONFIDENCE):
                width, height = int(detection[2] * wT), int(detection[3] * hT)
                x, y = int(detection[0] * wT - width / 2), int(detection[1] * hT - height / 2)
                boundingBox.append([x, y, width, height])
                classIds.append(classId)
                confidences.append(confidence)


    indices = cv2.dnn.NMSBoxes(boundingBox, confidences, CONFIDENCE, NMS_THRESHOLD)

    for i in indices:
        box = boundingBox[i]
        x, y, width, height = box[0], box[1], box[2], box[3]
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 2) # последние две: цвет, толстность
        cv2.putText(image, f'{classNames[classIds[i]].upper()} {int(confidences[i] * 100 % 100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) # font scale, color, thickness

# Загружаем кадры и сортируем их
input_frames_path = "./test/input/frames/"
path_names = []
for (path, _, filenames) in os.walk(input_frames_path):
    path_names.extend(os.path.join(path, name) for name in filenames)
path_names = [Path(i) for i in path_names]
path_names = sorted(path_names, key=lambda i: int(i.stem))
path_names = [str(i) for i in path_names]

sorted(path_names, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

coco_names_path = "./data/coco.names"
classNames = []
with open(coco_names_path, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

for path_name in path_names:
    # загрузка изображения
    image = cv2.imread(path_name)
    file_name = os.path.basename(path_name)
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1/255, (whT, whT), [0,0,0],crop=False)
    net.setInput(blob)

    # all layers
    layerNames = net.getLayerNames()
    # output layers
    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)

    findObjects(outputs, image)

    cv2.imshow('Image', image)
    cv2.waitKey(1)
