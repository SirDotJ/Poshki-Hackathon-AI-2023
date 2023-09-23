import cv2
import numpy
import numpy as np
import os
from pathlib import Path

CONFIDENCE = 0.5
NMS_THRESHOLD = 0.4
SCORE_THRESHOLD = 0.5
MINIMUM_WIDTH = 20
MINIMUM_HEIGHT = 40

whT = 320

# конфигурация нейронной сети
MODEL_CONFIGURATION_PATH = "./model/config/yolov3.cfg"
# файл весов сети YOLO
MODEL_WEIGHTS_PATH = "./model/weights/yolov3.weights"
# weights_path = "weights/yolov3-tiny.weights"
VIDEO_PATH = "./test/input/11_50_20.mp4"

# подготовка модели
net = cv2.dnn.readNetFromDarknet(MODEL_CONFIGURATION_PATH, MODEL_WEIGHTS_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
coco_names_path = "./data/coco.names"
classNames = []
with open(coco_names_path, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# детектирование на видео
capture = cv2.VideoCapture(VIDEO_PATH)
frameCounter = -1
while True:
    success, image = capture.read()
    if not success:
        break
    frameCounter += 1
    if frameCounter % 6 == 0:
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (whT, whT), [0, 0, 0], crop=False)
        net.setInput(blob)

        # all layers
        layerNames = net.getLayerNames()
        # output layers
        outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

        outputs = net.forward(outputNames)

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
                if (classId != 0): # проверяем только на наличие пешеходов
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
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 2)  # последние две: цвет, толстность
            cv2.putText(image, f'{classNames[classIds[i]].upper()} {int(confidences[i] * 100 % 100)}%', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # font scale, color, thickness

        cv2.imshow('Image', image)
        cv2.waitKey(1)

capture.release()
