import csv

from pathlib import Path
from PIL import Image

import numpy as np


def open_image(imagepath):
    if isinstance(imagepath, Path):
        imagepath = imagepath.resolve().as_posix()

    im = Image.open(imagepath, "r")
    im = im.convert("RGB")
    return im


def crop_to_bounding_box(image, bbox):
    x, y, w, h = bbox
    w = w + x
    h = y + h
    bbox = (x, y, w, h)
    return image.crop(bbox)


def restore_image_from_numpy(image):
    return Image.fromarray(np.uint8(image))


def read_csv(label_path):
    print(f"loading form {label_path}")
    with open(label_path) as csv_file:
        csv_file_rows = csv.reader(csv_file, delimiter=",")
        for row in csv_file_rows:
            yield row
