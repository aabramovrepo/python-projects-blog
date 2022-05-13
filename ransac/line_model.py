#
# RANSAC algorithm implementation for finding a linear model
#
# A hypothesis for a line model is made by picking two random points
# (a random subset of the original data), all other points are tested against
# the fitted model, the algorithm stops after a pre-defined number of
# iterations or once sufficiently many inliers are found.
#

import math
import os
import random
import sys
from dataclasses import dataclass

import click
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

SAMPLES_N = 500
STEPS_N = 20
THRESHOLD = 3
RATIO_INLIERS = 0.6
RATION_OUTLIERS = 0.4

DELTA = 10


@dataclass(frozen=True)
class Model:
    slope: float = 0.
    intercept: float = 0.


@dataclass(frozen=True)
class State:
    model: Model
    candidates: np.ndarray
    inliers: np.ndarray


def find_model(points: np.ndarray) -> Model:
    # warning: vertical and horizontal lines should be treated differently
    # here we just add some noise to avoid division by zero
    p1, p2 = points
    x1, y1 = p1
    x2, y2 = p2

    slope = (y2 - y1) / (x2 - x1 + sys.float_info.epsilon)
    intercept = y2 - slope * x2

    return Model(slope, intercept)


def find_intercept(model: Model, point: np.ndarray) -> np.ndarray:
    """find an intercept of a normal from the given point to the model"""
    x, y = point
    slope, intercept = model.slope, model.intercept
    eps = sys.float_info.epsilon

    x_ = (x + slope * y - slope * intercept) / ((1 + slope ** 2) + eps)
    y_ = (slope * x + (slope ** 2) * y - (slope ** 2) * intercept) / (
        (1 + slope ** 2) + eps) + intercept

    return np.asarray((x_, y_))


def generate_points(gauss_noise: bool = True) -> np.ndarray:
    # generate samples
    x = 30 * np.random.rand(SAMPLES_N, 1)

    # generate line's slope (called here perfect fit)
    y = x * random.uniform(-1, 1)

    # add a little gaussian noise
    x += np.random.normal(size=x.shape)
    y += np.random.normal(size=y.shape)

    # add some outliers to the point set
    n_outliers = int(RATION_OUTLIERS * SAMPLES_N)
    indices = np.arange(SAMPLES_N)
    np.random.shuffle(indices)
    outlier_indices = indices[:n_outliers]

    # gaussian outliers
    x[outlier_indices] = 30 * np.random.rand(n_outliers, 1)
    y[outlier_indices] = 30 * np.random.normal(size=(n_outliers, 1))

    # non-gaussian outliers (only on one side)
    if not gauss_noise:
        y[outlier_indices] = 30 * (np.random.normal(size=(n_outliers, 1)) ** 2)

    return np.hstack((x, y))


def plot_grid(points: np.ndarray) -> None:
    min_x, max_x = int(min(points[:, 0])), int(max(points[:, 0]))
    min_y, max_y = int(min(points[:, 1])), int(max(points[:, 1]))

    # grid for the plot
    grid = [min_x - DELTA, max_x + DELTA, min_y - DELTA, max_y + DELTA]
    plt.axis(grid)

    # put grid on the plot
    plt.grid(visible=True, which='major', color='0.75', linestyle='--')
    plt.xticks(range(min_x - DELTA, max_x + DELTA, 5))
    plt.yticks(range(min_y - DELTA, max_y + DELTA, 5))


def plot_points(points: np.ndarray, label: str, color: str,
                alpha: float) -> None:
    plt.plot(points[:, 0], points[:, 1], marker='o', label=label, color=color,
             linestyle='None', alpha=alpha)


def plot_line_model(points: np.ndarray, model: Model, color: str,
                    width: float) -> None:
    plt.plot(points[:, 0], model.slope * points[:, 0] + model.intercept,
             label='line model', color=color, linewidth=width)


def plot_progress(step: int, points: np.ndarray, state: State,
                  output_dir: str) -> None:
    plt.figure("RANSAC", figsize=(15., 15.))
    plot_grid(points)

    # plot input points
    plot_points(points, 'input points', '#00cc00', 0.4)

    # draw inliers
    if state.inliers.size:
        plot_points(state.inliers, 'inliers', '#ff0000', 0.6)

    # draw the current model
    plot_line_model(points, state.model, '#0080ff', 1.)

    # draw points picked up for the modeling
    plot_points(state.candidates, 'candidate points', '#0000cc', 0.6)

    plt.title('RANSAC iteration ' + str(step))
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'figure_' + str(step) + '.png'))
    plt.close()


def plot_results(points: np.ndarray, model: Model, output_dir: str) -> None:
    plt.figure("RANSAC", figsize=(15., 15.))
    plot_grid(points)
    plot_points(points, 'input points', '#00cc00', 0.4)
    plot_line_model(points, model, '#ff0000', 2.)

    plt.title('RANSAC final result')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'figure_final.png'))
    plt.close()


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)


def ransac_step(step: int, points: np.ndarray,
                output_dir: str) -> tuple[int, Model]:
    # pick up two random points
    indices = np.arange(SAMPLES_N)
    np.random.shuffle(indices)
    candidates, test_points = points[indices[:2]], points[indices[2:]]

    # find a line model for these points
    model = find_model(candidates)

    inliers = []
    # find orthogonal lines to the model for all testing points
    for point in test_points:
        # find an intercept of the model with a normal from the point
        intercept_point = find_intercept(model, point)

        # check whether step's an inlier or not
        if distance(intercept_point, point) < THRESHOLD:
            inliers.append(point)

    state = State(model, np.array(candidates), np.array(inliers))
    plot_progress(step, points, state, output_dir)

    return len(inliers), model


def run_ransac(points: np.ndarray, output_dir: str) -> Model:
    ratio = 0.
    model = Model()
    for step in tqdm(range(STEPS_N)):
        n_inliers, new_model = ransac_step(step, points, output_dir)
        new_ratio = n_inliers / SAMPLES_N
        if new_ratio > ratio:
            ratio = n_inliers / SAMPLES_N
            model = new_model

    return model


@click.command(help='Run RANSAC demo')
@click.argument('output_dir', type=click.Path(exists=True))
def main(output_dir: str) -> None:
    points = generate_points()
    model = run_ransac(points, output_dir)
    plot_results(points, model, output_dir)

    print(f'RANSAC steps: {STEPS_N}')
    print(f'Found model:\n slope: {model.slope:.3f}, '
          f'intercept: {model.intercept:.3f}')


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
