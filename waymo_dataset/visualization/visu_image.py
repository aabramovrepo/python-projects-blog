from pathlib import Path
from typing import cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from waymo_open_dataset.dataset_pb2 import CameraImage, CameraLabels

from visualization.colors import OBJECT_COLORS

Point = tuple[int, int]
Color = tuple[float, float, float, float]

CAMERA_NAME = {
    0: 'unknown',
    1: 'front',
    2: 'front-left',
    3: 'front-right',
    4: 'side-left',
    5: 'side-right'
}


def save_camera_image(idx: int, camera_image: CameraImage,
                      camera_labels: CameraLabels, output_dir: Path) -> None:
    image = tf.image.decode_png(camera_image.image)
    image = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)

    # draw the camera labels.
    for labels in camera_labels:
        if labels.name == camera_image.name:
            draw_labels(image, labels.labels)

    name = CAMERA_NAME[camera_image.name] + '-' + str(idx) + '.png'
    cv2.imwrite(str(output_dir / name), image)


def draw_labels(image: np.ndarray, labels: CameraLabels) -> None:
    def _draw_label(label_: CameraLabels) -> None:
        def _draw_line(p1: Point, p2: Point) -> None:
            cv2.line(image, p1, p2, color, 2)

        color = OBJECT_COLORS[label_.type]
        x1 = int(label_.box.center_x - 0.5 * label_.box.length)
        y1 = int(label_.box.center_y - 0.5 * label_.box.width)
        x2 = x1 + int(label_.box.length)
        y2 = y1 + int(label_.box.width)

        # draw bounding box
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
        for idx in range(len(points) - 1):
            _draw_line(points[idx], points[idx + 1])

    for label in labels:
        _draw_label(label)


def rgba_func(value: float) -> Color:
    """Generates a color based on a range value"""
    return cast(Color, plt.get_cmap('jet')(value / 50.))


def plot_points_on_image(idx: int, projected_points: np.ndarray,
                         camera_image: CameraImage, output_dir: Path) -> None:
    image = tf.image.decode_png(camera_image.image)
    image = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)

    for point in projected_points:
        x, y = int(point[0]), int(point[1])
        rgba = rgba_func(point[2])
        r, g, b = int(rgba[2] * 255.), int(rgba[1] * 255.), int(rgba[0] * 255.)
        cv2.circle(image, (x, y), 1, (b, g, r), 2)

    name = 'range-image-' + str(idx) + '-' + CAMERA_NAME[
        camera_image.name] + '.png'
    cv2.imwrite(str(output_dir / name), image)
