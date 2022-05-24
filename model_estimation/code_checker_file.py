import os
import click

COMMANDS = {'autopep8': 'autopep8 --in-place --recursive --max-line-length 79',
            'flake8': 'flake8',
            'isort': 'isort',
            'mypy': 'mypy',
            'pylint': 'pylint'}


@click.command()
@click.argument('file_path', type=click.Path(exists=True))
def main(file_path: str) -> None:
    for key, value in COMMANDS.items():
        print(f'---> CODE CHECK: {key}\n')
        print(file_path)
        command = value + ' ' + file_path
        os.system(command)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
