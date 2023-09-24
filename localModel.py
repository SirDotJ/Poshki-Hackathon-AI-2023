import time

import cv2
import numpy
import numpy as np
import os
from pathlib import Path
from shapely.geometry import Polygon
import winsound

import csv

CONFIDENCE = 0.6
NMS_THRESHOLD = 0.4
SCORE_THRESHOLD = 0.5
MINIMUM_WIDTH = 20
MINIMUM_HEIGHT = 40

whT = 320

END_DETECTION_KEY = 'q'

DANGER_ZONE_COORDINATES = np.array([[750, 575], [950, 575], [1300, 1080], [400, 1080], [750, 575]], np.int32)
DANGER_ZONE_COORDINATES.reshape((-1, 1, 2))
dangerZone = Polygon(DANGER_ZONE_COORDINATES)
DANGER_COOLDOWN = 4000
lastDangerTimestamp = -DANGER_COOLDOWN  # минус для учёта опасности в самом начале без случая
DANGER_MESSAGE_FRAME_LENGTH = 10
DANGER_MESSAGE_FRAME_COUNTER = 0

# конфигурация нейронной сети
MODEL_CONFIGURATION_PATH = "./model/config/yolov3.cfg"
# файл весов сети YOLO
MODEL_WEIGHTS_PATH = "./model/weights/yolov3.weights"
COCO_NAMES_PATH = "./model/coco.names"
# weights_path = "weights/yolov3-tiny.weights"
VIDEO_DIRECTORY_PATH = "./data/input/"
CSV_OUTPUT_PATH = "result.csv"

# подготовка модели
net = cv2.dnn.readNetFromDarknet(MODEL_CONFIGURATION_PATH, MODEL_WEIGHTS_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

classNames = []
with open(COCO_NAMES_PATH, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

testingResults = []

videoPaths = []
w = os.walk(VIDEO_DIRECTORY_PATH)
for(dirpath, dirnames, filenames) in w:
    for filename in filenames:
        videoPaths.append(VIDEO_DIRECTORY_PATH + filename)

for videoPath in videoPaths:

    # детектирование на видео
    capture = cv2.VideoCapture(videoPath)
    frameCounter = -1
    eventCounter = 0
    eventTimestamps = []
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

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
                    if (classId != 0):  # проверяем только на наличие пешеходов
                        continue
                    confidence = scores[classId]
                    if (confidence > CONFIDENCE):
                        width, height = int(detection[2] * wT), int(detection[3] * hT)
                        x, y = int(detection[0] * wT - width / 2), int(detection[1] * hT - height / 2)
                        boundingBox.append([x, y, width, height])
                        classIds.append(classId)
                        confidences.append(confidence)

            indices = cv2.dnn.NMSBoxes(boundingBox, confidences, CONFIDENCE, NMS_THRESHOLD)

            pedestrians = []
            for i in indices:
                box = boundingBox[i]
                x, y, width, height = box[0], box[1], box[2], box[3]
                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 2)  # последние две: цвет, толстность
                cv2.putText(image, f'{classNames[classIds[i]].upper()} {int(confidences[i] * 100 % 100)}%', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # font scale, color, thickness
                pedestrians.append(Polygon([(x, y), (x + width, y), (x + width, y + height), (x, y + height), (x, y)]))
            pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [DANGER_ZONE_COORDINATES], False, (50, 255, 255), 5)

            for pedestrian in pedestrians:
                if pedestrian.intersects(dangerZone):
                    if DANGER_MESSAGE_FRAME_COUNTER <= 0:
                        print("DANGER")

                        milliseconds = capture.get(cv2.CAP_PROP_POS_MSEC)
                        seconds = (int)(milliseconds / 1000)
                        secondsMessage = f'{seconds % 60}'

                        minutes = (int)(seconds / 60)
                        if(minutes < 10):
                            minuteMessage = f'0{minutes}'
                        else:
                            minuteMessage = f'{minutes}'

                        eventTimestamps.append(f'{minuteMessage}:{secondsMessage}')
                        eventCounter += 1

                        lastDangerTimestamp = time.time()
                        DANGER_MESSAGE_FRAME_COUNTER = DANGER_MESSAGE_FRAME_LENGTH

            if DANGER_MESSAGE_FRAME_COUNTER > 0:
                image = cv2.putText(image, "DANGER", (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
                winsound.Beep(300, 50)
                DANGER_MESSAGE_FRAME_COUNTER -= 1

            cv2.imshow('Image', image)
            cv2.waitKey(1)
    capture.release()

    # запись результата анализа в виде строки
    row = [videoPath, eventCounter]
    if eventCounter == 0:
        row.append('[]')
    else:
        node = ""
        node += f'[{eventTimestamps[0]}'
        for i in range(len(eventTimestamps)):
            if i == 0: # для избежания повторения первого случая
                continue
            node += f', {eventTimestamps[i]}'
        node += ']'
        node.strip()
        row.append(node)
    testingResults.append(row)

# сохранение результатов тестирования в выходную таблицу
header = ['filename', 'cases_count', 'timestamps']
with open (CSV_OUTPUT_PATH, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in testingResults:
        writer.writerow(row)
