import os
from pathlib import Path

import click
import numpy as np
import tensorflow as tf
from waymo_open_dataset.dataset_pb2 import Frame
from waymo_open_dataset.utils import frame_utils

from utils import save_frame, save_points
from visualization.visu_image import plot_points_on_image, save_camera_image
from visualization.visu_point_cloud import show_point_cloud

PcdList = list[np.ndarray]
PcdReturn = tuple[PcdList, PcdList]


def save_camera_images(idx: int, frame: Frame, output_dir: Path) -> None:
    for image in frame.images:
        save_camera_image(idx, image, frame.camera_labels, output_dir)


def save_data(frame: Frame, idx: int, points: np.ndarray,
              output_dir: Path) -> None:
    save_frame(frame, idx, output_dir)
    save_points(idx, points, output_dir)


def visualize_camera_projection(idx: int, frame: Frame, output_dir: Path,
                                pcd_return: PcdReturn) -> None:
    points, points_cp = pcd_return
    points_all = np.concatenate(points, axis=0)
    points_cp_all = np.concatenate(points_cp, axis=0)

    images = sorted(frame.images, key=lambda i: i.name)  # type: ignore

    # distance between lidar points and vehicle frame origin
    points_tensor = tf.norm(points_all, axis=-1, keepdims=True)
    points_cp_tensor = tf.constant(points_cp_all, dtype=tf.int32)

    mask = tf.equal(points_cp_tensor[..., 0], images[0].name)

    points_cp_tensor = tf.cast(tf.gather_nd(
        points_cp_tensor, tf.where(mask)), tf.float32)
    points_tensor = tf.gather_nd(points_tensor, tf.where(mask))

    projected_points_from_raw_data = tf.concat(
        [points_cp_tensor[..., 1:3], points_tensor], -1).numpy()

    plot_points_on_image(
        idx, projected_points_from_raw_data, images[0], output_dir)


def pcd_from_range_image(frame: Frame) -> tuple[PcdReturn, PcdReturn]:
    def _range_image_to_pcd(ri_index: int = 0) -> PcdReturn:
        points, points_cp = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose,
            ri_index=ri_index)
        return points, points_cp

    parsed_frame = frame_utils.parse_range_image_and_camera_projection(frame)
    range_images, camera_projections, _, range_image_top_pose = parsed_frame
    frame.lasers.sort(key=lambda laser: laser.name)
    return _range_image_to_pcd(), _range_image_to_pcd(1)


def visualize_pcd_return(frame: Frame, pcd_return: PcdReturn,
                         visu: bool) -> None:
    points, points_cp = pcd_return
    points_all = np.concatenate(points, axis=0)
    print(f'points_all shape: {points_all.shape}')

    # camera projection corresponding to each point
    points_cp_all = np.concatenate(points_cp, axis=0)
    print(f'points_cp_all shape: {points_cp_all.shape}')

    if visu:
        show_point_cloud(points_all, frame.laser_labels)


def concatenate_pcd_returns(
        pcd_return_1: PcdReturn,
        pcd_return_2: PcdReturn) -> tuple[np.ndarray, np.ndarray]:
    points, points_cp = pcd_return_1
    points_ri2, points_cp_ri2 = pcd_return_2
    points_concat = np.concatenate(points + points_ri2, axis=0)
    points_cp_concat = np.concatenate(points_cp + points_cp_ri2, axis=0)
    print(f'points_concat shape: {points_concat.shape}')
    print(f'points_cp_concat shape: {points_cp_concat.shape}')
    return points_concat, points_cp_concat


def process_data(idx: int, data: tf.Tensor, output_dir: Path, save: bool,
                 visu: bool) -> None:
    # pylint: disable=no-member (E1101)
    frame = Frame()
    frame.ParseFromString(bytearray(data.numpy()))

    # visualize point clouds of 1st and 2nd return
    pcd_return_1, pcd_return_2 = pcd_from_range_image(frame)
    visualize_pcd_return(frame, pcd_return_1, visu)
    visualize_pcd_return(frame, pcd_return_2, visu)

    # concatenate 1st and 2nd return
    points, _ = concatenate_pcd_returns(pcd_return_1, pcd_return_2)

    if visu:
        save_camera_images(idx, frame, output_dir)
        show_point_cloud(points, frame.laser_labels)
        visualize_camera_projection(idx, frame, output_dir, pcd_return_1)

    if save:
        save_data(frame, idx, points, output_dir)


def process_segment(segment_path: str, output_dir: Path, save: bool,
                    visu: bool) -> None:
    data_set = tf.data.TFRecordDataset(segment_path, compression_type='')
    for idx, data in enumerate(data_set):
        print(f'frame index: {idx}')
        process_data(idx, data, output_dir, save, visu)


@click.command(help='Point Cloud Visualization Demo')
@click.option('--save/--no-save', 'save', default=False,
              help='save frames and concatenated point clouds to disk')
@click.option('--visu/--no-visu', 'visu', default=False,
              help='visualize point clouds and save images')
@click.argument('segment_path', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=True))
def main(save: bool, visu: bool, segment_path: str, output_dir: str) -> None:
    if os.path.basename(segment_path).split('.')[-1] != 'tfrecord':
        raise ValueError(f'segment file has to be of '
                         f'{tf.data.TFRecordDataset.__name__} type')
    process_segment(segment_path, Path(output_dir), save, visu)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
