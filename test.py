import numpy as np
from matplotlib import pyplot as plt


def model(state, action):
    'model for motion'
    n, u = len(state), len(action)
    state_dot = np.zeros(n)
    speed, car_length = 1, 1
    state_dot[0] = action[0] * np.cos(state[2]) * speed
    state_dot[1] = action[0] * np.sin(state[2]) * speed
    state_dot[2] = np.tan(action[1]) *action[0]* speed / car_length
    state_dot[3] = action[1]

    return state_dot

def get_action_list():
    action_list = []
    max_speed = 5
    max_angle = 0.3
    angle = - max_angle
    speed = 0
    num_of_actions = 20
    while speed <= max_speed:
        while angle <= max_angle:
            action_list.append([speed, angle])
            angle += 2 * max_angle / num_of_actions
        speed += 2 * max_speed / num_of_actions
        angle = - max_angle
    return action_list

action_list = get_action_list()
from_node = [0, 0, 0, 0]
for action in action_list:
    current_node = from_node
    points = [current_node]
    for _ in range(100):
        current_node = current_node + 0.01 * model(current_node, np.array(action))
        points.append(current_node)
    points = np.array(points)
    plt.plot(points[:,0], points[:,1], '-')
    # print("current_node: ", current_node, "action:",action, "cost: ", cost)

plt.show()