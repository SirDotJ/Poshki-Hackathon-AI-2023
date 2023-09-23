import cv2
import os
from pathlib import Path

video_path = "./test/output/output.avi"
frame_folder = "./test/output/frames"

images = []
for (path, _, filenames) in os.walk(frame_folder):
    images.extend(os.path.join(path, name) for name in filenames)
images = [Path(i) for i in images]
images = sorted(images, key=lambda i: int(i.stem))
images = [str(i) for i in images]

firstImage = cv2.imread(images[0])
height, width = firstImage.shape[:2]

video = cv2.VideoWriter(video_path, 0, 6, (width, height))

for image in images:
    video.write(cv2.imread(image))

cv2.destroyAllWindows()
video.release()
