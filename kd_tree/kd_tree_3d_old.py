#
# K-D Tree (3D) construction and operations with it
#

import numpy as np
import math
import random
import pylab as plt
import mpl_toolkits.mplot3d.axes3d as p3

from collections import namedtuple
from operator import itemgetter
from pprint import pformat

# transparency level for drawing hyperplanes
plane_alpha = [.3, .25, .2, .15, .1]

nearest_nn = None  # nearest neighbor (NN)
distance_nn = float('inf')  # distance from NN to target


class Node(namedtuple('Node', 'location left_child right_child')):

    def __repr__(self):
        return pformat(tuple(self))


def generate_point_list_3d(n, min_val, max_val):
    """ generate a list of random 3D points
    :param n        number of points
    :param min_val  minimal value
    :return max_val maximal value
    """

    p = []

    for i in range(n):
        # coordinates as integer numbers
        # p.append((random.randint(min_val,max_val),
        # random.randint(min_val,max_val), random.randint(min_val,max_val)))

        # coordinates as real numbers
        p.append(
            (np.random.normal(random.randint(min_val, max_val), scale=0.5),
             np.random.normal(random.randint(min_val, max_val), scale=0.5),
             np.random.normal(random.randint(min_val, max_val), scale=0.5)))

    return p


def kdtree_3d(point_list, depth=0):
    """ build K-D tree
    :param point_list list of input points
    :param depth      current tree's depth
    :return tree node
    """

    try:
        k = len(point_list[0])  # assumes all points have the same dimension
    except IndexError:
        return None

    # Select axis based on depth so that axis cycles through all valid values
    axis = depth % k

    # Sort point list and choose median as pivot element
    point_list.sort(key=itemgetter(axis))
    median = len(point_list) // 2  # choose median

    # Create node and construct subtrees
    return Node(
        location=point_list[median],
        left_child=kdtree_3d(point_list[:median], depth + 1),
        right_child=kdtree_3d(point_list[median + 1:], depth + 1)
    )


def draw_sphere(ax, origin, radius, surface_color, surface_alpha):
    """ draw a sphere around the given point
    :param ax            3D axes
    :param origin        sphere's origin
    :param radius        sphere's radius
    :param surface_color color of sphere's surface
    :param surface_alpha alpha value for surface transparency
    """

    theta = np.linspace(0, 2 * np.pi, 20)
    phi = np.linspace(0, np.pi, 20)

    x_3d = origin[0] + radius * np.outer(np.cos(theta), np.sin(phi))
    y_3d = origin[1] + radius * np.outer(np.sin(theta), np.sin(phi))
    z_3d = origin[2] + radius * np.outer(np.ones(phi.shape), np.cos(phi))

    ax.plot_surface(x_3d, y_3d, z_3d, cstride=1, rstride=1,
                    color=surface_color, linewidth=0., alpha=surface_alpha)


def plot_tree_3d(tree, ax, min_x, max_x, min_y, max_y, min_z, max_z, prev_node,
                 branch, vis_planes=True, depth=0):
    """ draw 3D K-D tree
    :param tree K-D tree to be drawn
    :param ax   3D axes
    :param min_x
    :param max_x
    :param min_y
    :param max_y
    :param min_z
    :param max_z
    :param prev_node  parent's node
    :param branch     True if left, False if right
    :param vis_planes draw hyperplanes if True
    :param depth      depth level
    """

    cur_node = tree.location  # current tree's node
    left_branch = tree.left_child  # its left branch
    right_branch = tree.right_child  # its right branch

    # set line's width depending on tree's depth
    if depth > len(plane_alpha) - 1:
        pl_alpha = plane_alpha[len(plane_alpha) - 1]
    else:
        pl_alpha = plane_alpha[depth]

    k = len(cur_node)
    axis = depth % k

    # draw x splitting plane
    if axis == 0:

        if branch is not None and prev_node is not None:

            if branch:
                max_z = prev_node[2]
            else:
                min_z = prev_node[2]

        if vis_planes:
            y_grid = np.linspace(min_y, max_y, 10)
            z_grid = np.linspace(min_z, max_z, 10)
            yz_y, yz_z = plt.meshgrid(y_grid, z_grid)

            ax.plot_surface(cur_node[0], yz_y, yz_z, cstride=1, rstride=1,
                            color='#00cc00', linewidth=0.05, alpha=pl_alpha)

    # draw y splitting plane
    elif axis == 1:

        if branch is not None and prev_node is not None:

            if branch:
                max_x = prev_node[0]
            else:
                min_x = prev_node[0]

        if vis_planes:
            x_grid = np.linspace(min_x, max_x, 10)
            z_grid = np.linspace(min_z, max_z, 10)
            xz_x, xz_z = plt.meshgrid(x_grid, z_grid)

            ax.plot_surface(xz_x, cur_node[1], xz_z, cstride=1, rstride=1,
                            color='#0066ff', linewidth=0.05, alpha=pl_alpha)

    # draw z splitting plane
    elif axis == 2:

        if branch is not None and prev_node is not None:

            if branch:
                max_y = prev_node[1]
            else:
                min_y = prev_node[1]

        if vis_planes:
            x_grid = np.linspace(min_x, max_x, 10)
            y_grid = np.linspace(min_y, max_y, 10)
            xy_x, xy_y = plt.meshgrid(x_grid, y_grid)

            ax.plot_surface(xy_x, xy_y, cur_node[2], cstride=1, rstride=1,
                            color='#ff1a00', linewidth=0.05, alpha=pl_alpha)

    # draw the current node
    draw_sphere(ax, cur_node, 0.3, '#3366ff', 0.9)

    # draw left and right branches of the current node
    if left_branch is not None:
        plot_tree_3d(left_branch, ax, min_x, max_x, min_y, max_y, min_z, max_z,
                     cur_node, True, vis_planes, depth + 1)

    if right_branch is not None:
        plot_tree_3d(right_branch, ax, min_x, max_x, min_y, max_y, min_z,
                     max_z, cur_node, False, vis_planes, depth + 1)


def compute_closest_coordinate(value, range_min, range_max):
    """ Compute the closest coordinate for the neighboring hypercube
    :param value     coordinate value (x or y) of the target point
    :param range_min minimal coordinate (x or y) of the neighboring hypercube
    :param range_max maximal coordinate (x or y) of the neighboring hypercube
    :return x or y coordinate
    """

    v = None

    if range_min < value < range_max:
        v = value

    elif value <= range_min:
        v = range_min

    elif value >= range_max:
        v = range_max

    return v


def nearest_neighbor_search(tree, target_point, hr, distance, nearest=None,
                            depth=0):
    """ Find the nearest neighbor for the given point
    (claims O(log(n)) complexity)
    :param tree         K-D tree
    :param target_point given point for the NN search
    :param hr           splitting hypercube
    :param distance     minimal distance
    :param nearest      nearest point
    :param depth        tree's depth
    """

    global nearest_nn
    global distance_nn

    if tree is None:
        return

    k = len(target_point)

    cur_node = tree.location  # current tree's node
    left_branch = tree.left_child  # its left branch
    right_branch = tree.right_child  # its right branch

    nearer_kd = further_kd = None
    nearer_hr = further_hr = None
    left_hr = right_hr = None

    # Select axis based on depth so that axis cycles through all valid values
    axis = depth % k

    # split the hypercube depending on the axis
    if axis == 0:
        left_hr = [hr[0], (cur_node[0], hr[1][1], hr[1][2])]
        right_hr = [(cur_node[0], hr[0][1], hr[0][2]), hr[1]]

    if axis == 1:
        left_hr = [hr[0], (hr[1][0], cur_node[1], hr[1][2])]
        right_hr = [(hr[0][0], cur_node[1], hr[0][2]), hr[1]]

    if axis == 2:
        left_hr = [(hr[0][0], hr[0][1], cur_node[2]), hr[1]]
        right_hr = [hr[0], (hr[1][0], hr[1][1], cur_node[2])]

    # check which hypercube the target point belongs to
    if target_point[axis] <= cur_node[axis]:
        nearer_kd = left_branch
        further_kd = right_branch
        nearer_hr = left_hr
        further_hr = right_hr

    if target_point[axis] > cur_node[axis]:
        nearer_kd = right_branch
        further_kd = left_branch
        nearer_hr = right_hr
        further_hr = left_hr

    # check whether the current node is closer
    dist = (cur_node[0] - target_point[0]) ** 2 + (
            cur_node[1] - target_point[1]) ** 2 + (
                   cur_node[2] - target_point[2]) ** 2

    if dist < distance:
        nearest = cur_node
        distance = dist

    # go deeper in the tree
    nearest_neighbor_search(nearer_kd, target_point, nearer_hr, distance,
                            nearest, depth + 1)

    # once we reached the leaf node we check whether there are closer points
    # inside the hypersphere
    if distance < distance_nn:
        nearest_nn = nearest
        distance_nn = distance

    # a nearer point (px,py) could only be
    # in further_kd (further_hr) -> explore it
    px = compute_closest_coordinate(target_point[0], further_hr[0][0],
                                    further_hr[1][0])
    py = compute_closest_coordinate(target_point[1], further_hr[0][1],
                                    further_hr[1][1])
    pz = compute_closest_coordinate(target_point[2], further_hr[1][2],
                                    further_hr[0][2])

    # check whether it is closer than the current nearest neighbor =>
    # whether the hypersphere crosses the hypercube
    dist = (px - target_point[0]) ** 2 + (py - target_point[1]) ** 2 + (
            pz - target_point[2]) ** 2

    # explore the further kd-tree / hypercube if necessary
    if dist < distance_nn:
        nearest_neighbor_search(further_kd, target_point, further_hr, distance,
                                nearest, depth + 1)


def kd_tree_nn_search(step):
    """ Nearest neighbor search for a given input point
    :param step    index for the file name
    """
    n = 20  # number of points
    min_val = 0  # minimal coordinate value
    max_val = 30  # maximal coordinate value

    # generate list with input points
    point_list = generate_point_list_3d(n, min_val, max_val)
    print(point_list)

    # construct K-D tree
    kd_tree = kdtree_3d(point_list)

    # generate a target point

    # use normal distribution here !!!
    point = (
        random.randint(min_val, max_val), random.randint(min_val, max_val),
        random.randint(min_val, max_val))
    print(point)

    delta = 2  # extension of the drawing range
    hr = [(min_val - delta, min_val - delta, max_val + delta), (
        max_val + delta, max_val + delta,
        min_val - delta)]  # initial hyper cube

    visual = True  # True if results need to be plotted, False otherwise

    # create a figure for plotting data
    fig = plt.figure(figsize=(12., 12.))
    ax = p3.Axes3D(fig)

    # find the nearest neighbor
    max_dist = float('inf')
    nearest_neighbor_search(kd_tree, point, hr, max_dist)

    # draw the tree
    if visual:
        # True if hyperplanes need to be plotted, False otherwise
        vis_planes = True

        plot_tree_3d(kd_tree, ax, min_val - delta, max_val + delta,
                     min_val - delta, max_val + delta, min_val - delta,
                     max_val + delta, None, None, vis_planes)

        # draw the given point
        draw_sphere(ax, point, 0.3, '#f5003d', 0.9)

        # draw the hypersphere around the target point
        draw_sphere(ax, point, math.sqrt(distance_nn), '#ffff00', 0.3)

        # draw the found nearest neighbor
        draw_sphere(ax, nearest_nn, 0.5, '#006600', 0.9)

    # straightforward search in the list for a quality check
    min_distance = float('inf')
    min_distance_2 = float('inf')
    nn_point = None

    for ind in range(len(point_list)):

        dist = math.sqrt((point_list[ind][0] - point[0]) ** 2 + (
                point_list[ind][1] - point[1]) ** 2 + (
                                 point_list[ind][2] - point[2]) ** 2)
        dist_2 = (point_list[ind][0] - point[0]) ** 2 + (
                point_list[ind][1] - point[1]) ** 2 + (
                         point_list[ind][2] - point[2]) ** 2

        if dist < min_distance:
            min_distance = dist
            min_distance_2 = dist_2
            nn_point = point_list[ind]

    if nearest_nn[0] != nn_point[0] or nearest_nn[1] != nn_point[1] or \
            nearest_nn[2] != nn_point[2]:

        if min_distance_2 < distance_nn:
            print('\n NN search mismatch !!! \n')
            print('step = {}'.format(step))
            print('K-D tree NN: {}'.format(nearest_nn))
            print('List NN: {}'.format(nn_point))
            print('point = {}'.format(point))
            print('min_distance_2 = {}'.format(min_distance_2))
            print('distance_nn = {}'.format(distance_nn))
            print('point_list = {}'.format(point_list))

        elif min_distance_2 == distance_nn:
            print('step = {}'.format(step))
            print('Different nodes, but the same distance!')

        # in case of the mismatch draw the correct NN
        if visual:
            draw_sphere(ax, nn_point, 0.5, '#cc3366', 0.9)

    if visual:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim(min_val - delta, max_val + delta)
        ax.set_ylim(min_val - delta, max_val + delta)
        ax.set_zlim(min_val - delta, max_val + delta)

        fig.add_axes(ax)
        plt.title('K-D Tree in 3D')
        plt.show()
        # plt.savefig('output/K-D-Tree_NN_Search_3D_' + str(step) + '.png')

    # plt.close()


def main():
    global nearest_nn
    global distance_nn

    kd_tree_nn_search(0)


#    for i in range(100):
#        nearest_nn = None
#        distance_nn = float('inf')
#        kd_tree_nn_search(i)

if __name__ == '__main__':
    main()
