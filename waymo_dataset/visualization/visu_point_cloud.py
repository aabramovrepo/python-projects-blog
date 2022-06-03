import numpy as np
import open3d as o3d
from waymo_open_dataset.label_pb2 import Label
from waymo_open_dataset.utils import transform_utils

Point3D = list[float]
LineSegment = tuple[int, int]

# order in which bbox vertices will be connected
LINE_SEGMENTS = [[0, 1], [1, 3], [3, 2], [2, 0],
                 [4, 5], [5, 7], [7, 6], [6, 4],
                 [0, 4], [1, 5], [2, 6], [3, 7]]


def get_bbox(label: Label) -> np.ndarray:
    width, length = label.box.width, label.box.length
    return np.array([[-0.5 * length, -0.5 * width],
                     [-0.5 * length, 0.5 * width],
                     [0.5 * length, -0.5 * width],
                     [0.5 * length, 0.5 * width]])


def transform_bbox_waymo(label: Label) -> np.ndarray:
    """Transform object's 3D bounding box using Waymo utils"""
    heading = -label.box.heading
    bbox_corners = get_bbox(label)

    mat = transform_utils.get_yaw_rotation(heading)
    rot_mat = mat.numpy()[:2, :2]

    return bbox_corners @ rot_mat


def transform_bbox_custom(label: Label) -> np.ndarray:
    """Transform object's 3D bounding box without Waymo utils"""
    heading = -label.box.heading
    bbox_corners = get_bbox(label)
    rot_mat = np.array([[np.cos(heading), - np.sin(heading)],
                        [np.sin(heading), np.cos(heading)]])

    return bbox_corners @ rot_mat


def build_open3d_bbox(box: np.ndarray, label: Label) -> list[Point3D]:
    """Create bounding box's points and lines needed for drawing in open3d"""
    x, y, z = label.box.center_x, label.box.center_y, label.box.center_z

    z_bottom = z - label.box.height / 2
    z_top = z + label.box.height / 2

    points = [[0., 0., 0.]] * box.shape[0] * 2
    for idx in range(box.shape[0]):
        x_, y_ = x + box[idx][0], y + box[idx][1]
        points[idx] = [x_, y_, z_bottom]
        points[idx + 4] = [x_, y_, z_top]

    return points


def show_point_cloud(points: np.ndarray, laser_labels: Label) -> None:
    # pylint: disable=no-member (E1101)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.6, origin=[0, 0, 0])

    vis.add_geometry(pcd)
    vis.add_geometry(mesh_frame)

    for label in laser_labels:
        bbox_corners = transform_bbox_waymo(label)
        # bbox_corners = transform_bbox_custom(label)
        bbox_points = build_open3d_bbox(bbox_corners, label)

        colors = [[1, 0, 0] for _ in range(len(LINE_SEGMENTS))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(bbox_points),
            lines=o3d.utility.Vector2iVector(LINE_SEGMENTS),
        )

        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)

    vis.run()
