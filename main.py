# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
import time
import sys
import os

CONFIDENCE = 0.4
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
MINIMUM_WIDTH = 20
MINIMUM_HEIGHT = 40

# конфигурация нейронной сети
config_path = "C:\\Users\\sirdo\\PycharmProjects\\Poshki-Hackathon-AI-2023\\model\\config\\yolov3.cfg"
# файл весов сети YOLO
weights_path = "C:\\Users\\sirdo\\PycharmProjects\\Poshki-Hackathon-AI-2023\\model\\weights\\yolov3.weights"
# weights_path = "weights/yolov3-tiny.weights"
path_names = [
    "C:\\Users\\sirdo\\PycharmProjects\\Poshki-Hackathon-AI-2023\\test\\001-bad",
    "C:\\Users\\sirdo\\PycharmProjects\\Poshki-Hackathon-AI-2023\\test\\002-bad.jpg",
    "C:\\Users\\sirdo\\PycharmProjects\\Poshki-Hackathon-AI-2023\\test\\003-bad.jpg",
    "C:\\Users\\sirdo\\PycharmProjects\\Poshki-Hackathon-AI-2023\\test\\004-bad.jpg"
]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # разделение видео на отдельные кадры



    # загрузка всех меток классов (объектов)
    labels = open("data/coco.names").read().strip().split("\n")
    # генерируем цвета для каждого объекта и последующего построения
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
    # загружаем сеть YOLO
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    for path_name in path_names:

        # загрузка изображения
        image = cv2.imread(path_name)
        file_name = os.path.basename(path_name)
        height, width = image.shape[:2]

        # нормализация изображения
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (1024, 1024), swapRB=True, crop=False)

        # получение прогнозов
        net.setInput(blob)
        layerNames = net.getLayerNames()
        layerNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
        start = time.perf_counter()
        layer_outputs = net.forward(layerNames)
        time_took = time.perf_counter() - start
        print(f"Потребовалось: {time_took:.2f}s")

        # фильтруем то что больше CONFIDENCE
        font_scale = 1
        thickness = 1
        boxes, confidences, class_ids = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > CONFIDENCE:
                    box = (detection[:4] * np.array([width, height, width, height]))
                    (centerX, centerY, width, height) = box.astype("int")
                    if width >= MINIMUM_WIDTH and height >= MINIMUM_HEIGHT:
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        # отрисовка обнаруженных объектов в файл
        for i in range(len(boxes)):
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = image.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
        cv2.imwrite(file_name + "_yolo3.png", image)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
