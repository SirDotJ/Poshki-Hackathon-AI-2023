import cv2
import numpy
import numpy as np
import os
from pathlib import Path
from roboflow import Roboflow

# Параметры программы
ROBOFLOW_API_KEY = "vOKeVrZ9227xTYal9FFO"
ROBOFLOW_PROJECT_NAME = "trains-qjwao"
INPUT_FRAMES_PATH = "./test/input/frames/"
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

# Загружаем кадры и сортируем их
path_names = []
for (path, _, filenames) in os.walk(INPUT_FRAMES_PATH):
    path_names.extend(os.path.join(path, name) for name in filenames)
path_names = [Path(i) for i in path_names]
path_names = sorted(path_names, key=lambda i: int(i.stem))
path_names = [str(i) for i in path_names]
sorted(path_names, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

# Проходим через каждый кадр
for path_name in path_names:
    # Загружаем изображение
    image = cv2.imread(path_name)
    boundingBoxes = []
    classNames = []
    confidences = []

    # Получаем предсказания
    detections = model.predict(path_name, confidence=CONFIDENCE, overlap=NMS_THRESHOLD)

    for bounding_box in detections:
        # Ищем найденные
        x = bounding_box['x']
        y = bounding_box['y']
        width = bounding_box['width']
        height = bounding_box['height']
        confidence = bounding_box['confidence']
        className = bounding_box['class']
        x = (int)(x - width / 2)
        y = (int)(y - height / 2)
        if (confidence > CONFIDENCE):
            boundingBoxes.append((x, y, width, height))
            classNames.append(className)
            confidences.append(confidence)

    indices = cv2.dnn.NMSBoxes(boundingBoxes, confidences, CONFIDENCE, NMS_THRESHOLD)
    for i in indices:
        box = boundingBoxes[i]
        x, y, width, height = box[0], box[1], box[2], box[3]
        className = classNames[i]
        if(className == "trail"):
            color = TRAIL_COLOR
        else:
            color = PERSON_COLOR
        cv2.rectangle(image, (x, y), (x + width, y + height), color, THICKNESS)
        cv2.putText(image, f'{className.upper()} {int(confidences[i] * 100 % 100)}%',
                    (x, y - 10), FONT, FONT_SCALE, color, THICKNESS)

    cv2.imshow('Image', image)
    cv2.waitKey(1)

