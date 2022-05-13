import click
from PIL import Image


@click.command(help='Create image with segmentation overlay')
@click.argument('image_path', type=click.Path(exists=True))
@click.argument('segments_png_path', type=click.Path(exists=True))
def main(image_path: str, segments_png_path: str) -> None:
    image = Image.open(image_path)
    segments = Image.open(segments_png_path)

    map_mesh = Image.new('RGBA', image.size, (0, 0, 0, 100))
    image.paste(segments, (0, 0), map_mesh)
    image.save('output.png')


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
