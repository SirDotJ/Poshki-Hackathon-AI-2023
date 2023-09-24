import cv2
import numpy
import numpy as np
import os
from pathlib import Path
from roboflow import Roboflow

# Параметры программы
ROBOFLOW_API_KEY = "vOKeVrZ9227xTYal9FFO"
ROBOFLOW_PROJECT_NAME = "trains-qjwao"
VIDEO_PATH = "./data/input/11_50_20.mp4"
WORKING_FRAME_DIRECTORY = "./data/output/frames"
CONFIDENCE = 0.4
NMS_THRESHOLD = 0.4
TRAIL_COLOR = (255, 0, 140)
PERSON_COLOR = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2 # прямоугольника и текста

# Включаем модель Roboflow
roboflow = Roboflow(api_key=ROBOFLOW_API_KEY)
project = roboflow.workspace().project(ROBOFLOW_PROJECT_NAME)
model = project.version(2).model

capture = cv2.VideoCapture(VIDEO_PATH)
frameCounter = -1
while True:
    success, image = capture.read()
    if not success:
        break
    frameCounter = frameCounter + 1
    if frameCounter % 48 == 0:
        # Сохраняем изображение для отправки в API
        cv2.imwrite(f'{WORKING_FRAME_DIRECTORY}frame_{frameCounter}.jpg', image)
        temporaryImagePath = f'{WORKING_FRAME_DIRECTORY}frame_{frameCounter}.jpg'

        # Массивы данных результатов детектирования
        boundingBoxes = []
        classNames = []
        confidences = []

        # Получаем детектирования с API
        detections = model.predict(temporaryImagePath, confidence=CONFIDENCE, overlap=NMS_THRESHOLD)

        # Проходим через все прямоугольники детектирования
        for bounding_box in detections:
            x = bounding_box['x']
            y = bounding_box['y']
            width = bounding_box['width']
            height = bounding_box['height']
            confidence = bounding_box['confidence']
            className = bounding_box['class']
            # Преобразования для учёта: RoboFlow отмечает серединную координату
            x = (int)(x - width / 2)
            y = (int)(y - height / 2)
            if (confidence > CONFIDENCE):
                boundingBoxes.append((x, y, width, height))
                classNames.append(className)
                confidences.append(confidence)

        # Отсекаем пересекающие и помещаем результат на изображение
        indices = cv2.dnn.NMSBoxes(boundingBoxes, confidences, CONFIDENCE, NMS_THRESHOLD)
        for i in indices:
            box = boundingBoxes[i]
            x, y, width, height = box[0], box[1], box[2], box[3]
            className = classNames[i]
            if (className == "trail"):
                color = TRAIL_COLOR
            else:
                color = PERSON_COLOR
            cv2.rectangle(image, (x, y), (x + width, y + height), color, THICKNESS)
            cv2.putText(image, f'{className.upper()} {int(confidences[i] * 100 % 100)}%',
                        (x, y - 10), FONT, FONT_SCALE, color, THICKNESS)

        # Вывод изображения и реинициализация перед следующим кадром
        cv2.imshow('Image', image)
        os.remove(temporaryImagePath)
        cv2.waitKey(1)
capture.release()