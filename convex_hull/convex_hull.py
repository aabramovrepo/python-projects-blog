#
# The current code contains implementation of Andrew's monotone chain
# 2D convex hull algorithm. It's asymptotic complexity: O(n log n).
# Practical performance: 0.5-1.0 seconds for n=1000000 on a 1GHz machine.
#
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt

POINTS_N = 15  # number of points

GRID_WIDTH = 10
GRID_HEIGHT = 10


@dataclass(frozen=True, order=True)
class Point:
    x: int = 0
    y: int = 0


Points = list[Point]


def generate_point() -> Point:
    return Point(random.randint(0, GRID_WIDTH), random.randint(0, GRID_HEIGHT))


def draw_grid() -> None:
    plt.axis([-1, GRID_WIDTH + 1, -1, GRID_HEIGHT + 1])
    plt.grid(visible=True, which='major', color='0.75', linestyle='--')
    plt.xticks(range(-5, GRID_WIDTH + 1, 1))
    plt.yticks(range(-1, GRID_HEIGHT + 5, 1))


def draw_convex_hull(
        points: Points, lower_hull: Points, upper_hull: Points) -> None:
    plt.figure('Convex hull computation')
    draw_grid()
    plt.plot([p.x for p in points], [p.y for p in points], 'ko')
    plt.plot([p.x for p in lower_hull], [p.y for p in lower_hull],
             linestyle='-', color='blue', label='lower hull')
    plt.plot([p.x for p in upper_hull], [p.y for p in upper_hull],
             linestyle='-', color='red', label='upper hull')
    plt.legend(['Points', 'Lower Hull', 'Upper Hull'], loc='upper left')
    plt.show()
    plt.close()


def draw_results(convex_hull: Points, point: Point, is_inside: bool) -> None:
    fig = plt.figure('Checking a point')
    ax = fig.add_subplot()
    draw_grid()
    ax.text(
        0.95, 0.01, 'Inside' if is_inside else 'Outside',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, color='green', fontsize=15)

    plt.plot([p.x for p in convex_hull], [p.y for p in convex_hull],
             linestyle='-', color='blue')
    plt.plot(point.x, point.y, 'go')
    plt.legend(['Convex Hull', 'Input Point'], loc='upper left')
    plt.show()
    plt.close()


def cross(point_o: Point, point_a: Point, point_b: Point) -> int:
    """ 2D cross product of OA and OB vectors,
    i.e. z-component of their 3D cross product
    :param point_o: point O
    :param point_a: point A
    :param point_b: point B
    :return cross product of vectors OA and OB (OA x OB),
    positive if OAB makes a counter-clockwise turn,
    negative for clockwise turn, and zero if the points are collinear
    """
    return (point_a.x - point_o.x) * (point_b.y - point_o.y) - (
        point_a.y - point_o.y) * (point_b.x - point_o.x)


def compute_hull_side(points: Points) -> Points:
    hull: Points = []
    for p in points:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    return hull


def compute_convex_hull(points: Points) -> Points:
    """ Computation of a convex hull for a set of 2D points.
    :param points: sequence of points
    :return a list of vertices of the convex hull in counter-clockwise order,
    starting from the vertex with the lexicographically smallest coordinates
    """
    # remove duplicates and sort points lexicographically
    # to detect the case we have just one unique point
    points = sorted(list(set(points)))

    # boring case: no points or a single point,
    # possibly repeated multiple times
    if len(points) <= 1:
        return points

    # build lower and upper hulls
    lower_hull = compute_hull_side(points)
    upper_hull = compute_hull_side(list(reversed(points)))

    # concatenation of the lower and upper hulls gives the convex hull;
    # the first point occurs in the list twice,
    # since it's at the same time the last point
    convex_hull = lower_hull + upper_hull
    draw_convex_hull(points, lower_hull, upper_hull)

    return convex_hull


def check_point(convex_hull: Points, point: Point) -> None:
    def _is_inside() -> bool:
        for idx in range(1, len(convex_hull)):
            if cross(convex_hull[idx - 1], convex_hull[idx], point) < 0:
                return False
        return True

    # visualize results
    draw_results(convex_hull, point, _is_inside())


def main() -> None:
    # generate input points
    points = [generate_point() for _ in range(POINTS_N)]

    # find a convex hull
    convex_hull = compute_convex_hull(points)

    # generate a point to be checked
    point = generate_point()

    # check the point
    check_point(convex_hull, point)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
