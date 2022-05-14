from pathlib import Path

import numpy as np
from waymo_open_dataset import dataset_pb2


def save_frame(frame: dataset_pb2.Frame, idx: int, output_dir: Path) -> None:
    name = 'frame-' + str(idx) + '.bin'
    with open((output_dir / name), 'wb') as file:
        file.write(frame.SerializeToString())


def save_points(idx: int, points: np.ndarray, output_dir: Path) -> None:
    name = 'points-' + str(idx) + '.npy'
    np.save(str(output_dir / name), points)
