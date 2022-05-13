import random

import click
import numpy as np
from PIL import Image

SEGMENTS_N = 3000

Color = tuple[int, int, int]


def generate_colors() -> list[Color]:
    def _random_number() -> int:
        return random.randint(0, 255)

    def _random_color() -> Color:
        return _random_number(), _random_number(), _random_number()

    return [_random_color() for _ in range(SEGMENTS_N)]


def colored_segments(segments: np.ndarray, colors: list[Color]) -> np.ndarray:
    height, width = segments.shape
    segments_image = np.zeros([height, width, 3], dtype=np.uint8)
    segment_ids = np.unique(segments)
    for id_ in segment_ids:
        segments_image[:, :, :][(segments == id_)] = colors[id_]

    return segments_image


@click.command(help='Create image with segmentation overlay')
@click.argument('image_path', type=click.Path(exists=True))
@click.argument('segments_npy_path', type=click.Path(exists=True))
def main(image_path: str, segments_npy_path: str) -> None:
    image = Image.open(image_path)
    segments = np.load(segments_npy_path)

    colors = generate_colors()
    segments_mask = colored_segments(segments, colors)
    segments_mask = Image.fromarray(np.uint8(segments_mask))

    map_mesh = Image.new('RGBA', image.size, (0, 0, 0, 100))
    image.paste(segments_mask, (0, 0), map_mesh)
    image.save('output.png')


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
