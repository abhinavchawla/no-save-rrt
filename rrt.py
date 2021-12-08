import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import random
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.path import Path
import time


class Artists:
    'artists for animating tree search'

    def __init__(self, ax):
        self.artist_list = []
        self.ax = ax
        self.rand_pt_marker, = ax.plot([], [], '--o', color='lime', lw=1, zorder=1)
        self.artist_list.append(self.rand_pt_marker)

        self.goal_pt_marker, = ax.plot([], [], '--o', color='red', lw=1, zorder=2)
        self.artist_list.append(self.goal_pt_marker)

        self.root_pt_marker, = ax.plot([], [], '--o', color='blue', lw=1, zorder=2)
        self.artist_list.append(self.root_pt_marker)

        self.obs_solid_lines = LineCollection([], lw=2, animated=True, color='k', zorder=1)
        ax.add_collection(self.obs_solid_lines)
        self.artist_list.append(self.obs_solid_lines)

        self.path_to_goal_lines = LineCollection([], lw=2, animated=True, color='blue', zorder=1)
        ax.add_collection(self.path_to_goal_lines)
        self.artist_list.append(self.path_to_goal_lines)

    def update_rand_pt_marker(self, rand_pt):
        'update random point marker'

        xs = [rand_pt[0]]
        ys = [rand_pt[1]]

        self.rand_pt_marker.set_data(xs, ys)

    def update_goal_pt_marker(self, goal_pt):
        'update goal point marker'

        xs = [goal_pt[0]]
        ys = [goal_pt[1]]

        self.goal_pt_marker.set_data(xs, ys)

    def update_root_pt_marker(self, root_pt):
        'update root point marker'

        xs = [root_pt[0]]
        ys = [root_pt[1]]

        self.root_pt_marker.set_data(xs, ys)

    def update_obs_solid_lines(self, old_pt, new_pt):
        codes = [Path.MOVETO, Path.LINETO]
        verts = [(new_pt.pos[0], new_pt.pos[1]), (old_pt.pos[0], old_pt.pos[1])]
        obs_solid_paths = self.obs_solid_lines.get_paths()
        obs_solid_paths.append(Path(verts, codes))

    def update_circles(self, obstacles):
        for obs in obstacles:
            circle1 = plt.Circle((obs[0], obs[1]), 0.05, color='r')
            self.ax.add_patch(circle1)
            self.artist_list.append(circle1)

    def update_path_to_goal(self, path):
        'update artist list'
        for i in range(len(path) - 1):
            codes = [Path.MOVETO, Path.LINETO]
            verts = [(path[i][0], path[i][1]), (path[i + 1][0], path[i + 1][1])]
            path_to_goal_paths = self.path_to_goal_lines.get_paths()
            path_to_goal_paths.append(Path(verts, codes))


class TreeNode:
    def __init__(self, pos, parent):
        self.pos = pos
        self.parent = parent
        self.children = []
        self.cost = 0
        self.path_cost = 0
        self.total_cost = 0


class RRT:
    'RRT algorithm'

    def __init__(self, start, goal, obstacle_list, rand_area, step_size, max_iter, filename=None):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacle_list = obstacle_list
        self.rand_area = rand_area
        self.step_size = step_size
        self.max_iter = max_iter

        self.path = []
        self.path_found = False
        self.root = TreeNode(self.start, None)

        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
        self.artists = Artists(self.ax)
        self.node_list = [self.root]
        self.anim = None
        self.artists.update_root_pt_marker(self.start)
        self.artists.update_goal_pt_marker(self.goal)
        self.artists.update_circles(self.obstacle_list)
        self.i = 0
        self.filename = filename
        print(self.filename)

    def get_random_point(self):
        'get random point in random area'

        x_min, x_max = self.rand_area[0]
        y_min, y_max = self.rand_area[1]

        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)

        return np.array([x, y])

    def get_nearest_node(self, node):
        'get nearest node in tree'

        min_dist = float('inf')
        nearest_node = None

        for n in self.node_list:
            dist = np.linalg.norm(node - n.pos)

            if dist < min_dist:
                min_dist = dist
                nearest_node = n

        return nearest_node

    def steer(self, from_node, to_node):
        'steer from from_node to to_node'

        theta = np.arctan2(to_node[1] - from_node.pos[1], to_node[0] - from_node.pos[0])
        dist = np.linalg.norm(to_node - from_node.pos)

        if dist < self.step_size:
            new_node = TreeNode(to_node, from_node)
        else:
            new_node = TreeNode(from_node.pos + self.step_size * np.array([np.cos(theta), np.sin(theta)]), from_node)

        if self.collision_check(new_node):
            from_node.children.append(new_node)
            self.node_list.append(new_node)
            return new_node
        else:
            return None

    def collision_check(self, node):
        'check if node is in collision'

        for obs in self.obstacle_list:
            if np.linalg.norm(node.pos - obs) < 0.05:
                return False
        return True

    def iterate(self):
        'iterate RRT algorithm'
        random_pt = self.get_random_point()
        nearest_node = self.get_nearest_node(random_pt)
        new_node = self.steer(nearest_node, random_pt)
        if new_node:
            dist = self.get_dist(new_node, self.goal)
            if dist < 0.03:
                self.update_path(new_node)
        return random_pt, nearest_node, new_node

    def update_path(self, new_node):
        self.path_found = True
        self.path = self.get_path(new_node)
        self.path_length = len(self.path)
        self.time_taken = time.time() - self.start_time
        self.total_nodes = len(self.node_list)
        print('path found ', self.path_length)
        print('Time taken: ', self.time_taken)
        print('Total Nodes explored', self.total_nodes)
        self.artists.update_path_to_goal(self.path)

    def return_results(self):
        return self.path_found, self.path_length, self.time_taken, self.total_nodes

    def get_dist(self, node, goal):
        return np.linalg.norm(node.pos - goal)

    def get_path(self, node):
        'get path from root to node'

        path = [node.pos]

        while node.parent is not None:
            node = node.parent
            path.append(node.pos)

        return path[::-1]

    def animate(self, i):
        'animation function'
        if not self.path_found:
            # print('iteration: ', i)
            random_pt, nearest_node, new_node = self.iterate()
            self.artists.update_rand_pt_marker(random_pt)
            if new_node and nearest_node:
                self.artists.update_obs_solid_lines(nearest_node, new_node)
        return self.artists.artist_list

    def run(self):
        'run RRT algorithm'
        # plot root point (not animated)
        self.ax.plot([self.root.pos[0]], [self.root.pos[1]], 'ko', ms=5)
        self.start_time = time.time()
        self.anim = animation.FuncAnimation(self.fig, self.animate, frames=self.max_iter, interval=1, blit=True)
        self.anim.save(self.filename, fps=30, writer='imagemagick')


class RRT_Opt(RRT):

    def __init__(self, start, goal, obstacle_list, rand_area, step_size, max_iter, filename=None):
        super().__init__(start, goal, obstacle_list, rand_area, step_size, max_iter, filename)
        self.current_rand_pt = None
        self.old_node = self.root

    def iterate(self):
        if self.current_rand_pt is None or self.current_node is None or self.get_dist(self.current_node,
                                                                                      self.current_rand_pt) < 0.001:
            self.current_rand_pt = self.get_random_point()
            self.current_node = self.get_nearest_node(self.current_rand_pt)
        self.old_node = self.current_node
        self.current_node = self.steer(self.current_node, self.current_rand_pt)

        if self.current_node:
            dist = self.get_dist(self.current_node, self.goal)
            if dist < 0.03:
                self.update_path(self.current_node)

        return self.current_rand_pt, self.current_node, self.old_node


def generate_random_point(search_space):
    x_min, x_max = search_space[0]
    y_min, y_max = search_space[1]

    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)

    return np.array([x, y])


def collision_check(node, obstacle_list):
    'check if node is in collision'

    for obs in obstacle_list:
        if np.linalg.norm(node - obs) < 0.05:
            return False
    return True


if __name__ == '__main__':
    search_space = np.array([[0, 1], [0, 1]])
    obstacles = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9],
                 (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.3, 0.7), (0.2, 0.8),
                 (0.1, 0.9)]
    path_found_rrt = []
    path_lengths_rrt = []
    time_taken_rrt = []
    total_nodes_rrt = []

    path_found_rrt_opt = []
    path_lengths_rrt_opt = []
    time_taken_rrt_opt = []
    total_nodes_rrt_opt = []
    for test in range(10):
        start = generate_random_point(search_space)
        goal = generate_random_point(search_space)
        while not collision_check(start, obstacles):
            print('start in collision')
            start = generate_random_point(search_space)
        while not collision_check(goal, obstacles):
            print('goal in collision')
            goal = generate_random_point(search_space)
        print('Distance between start and goal ', np.linalg.norm(start - goal))
        rrtsearch = RRT(start, goal, obstacles, search_space, 0.01, 5000, filename='cache/rrt_' + str(test) + '.gif')
        rrtsearch.run()

        results = rrtsearch.return_results()
        path_found_rrt.append(results[0])
        path_lengths_rrt.append(results[1])
        time_taken_rrt.append(results[2])
        total_nodes_rrt.append(results[3])

        rrtsearch2 = RRT_Opt(start, goal, obstacles, search_space, 0.01, 5000,
                             filename='cache/rrt_opt_' + str(test) + '.gif')
        rrtsearch2.run()

        results = rrtsearch2.return_results()
        path_found_rrt_opt.append(results[0])
        path_lengths_rrt_opt.append(results[1])
        time_taken_rrt_opt.append(results[2])
        total_nodes_rrt_opt.append(results[3])
        print('test ', test)
        print('path found: ', path_found_rrt)
        print('path lengths: ', path_lengths_rrt)
        print('time taken: ', time_taken_rrt)
        print('total nodes: ', total_nodes_rrt)

        print('path found: ', path_found_rrt_opt)
        print('path lengths: ', path_lengths_rrt_opt)
        print('time taken: ', time_taken_rrt_opt)
        print('total nodes: ', total_nodes_rrt_opt)
