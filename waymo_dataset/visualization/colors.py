# pylint: disable=no-member (E1101)
from dataclasses import dataclass

from waymo_open_dataset import label_pb2

Color = tuple[int, int, int]


@dataclass(frozen=True)
class ColorCodes:
    vehicle: Color = (0, 0, 255)
    pedestrian: Color = (0, 255, 0)
    cyclist: Color = (0, 255, 255)
    sign: Color = (255, 0, 0)


OBJECT_COLORS = {
    label_pb2.Label.Type.TYPE_VEHICLE: ColorCodes.vehicle,
    label_pb2.Label.Type.TYPE_PEDESTRIAN: ColorCodes.pedestrian,
    label_pb2.Label.Type.TYPE_CYCLIST: ColorCodes.cyclist,
    label_pb2.Label.Type.TYPE_SIGN: ColorCodes.sign
}
