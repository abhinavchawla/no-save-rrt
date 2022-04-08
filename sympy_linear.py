import numpy as np
from sympy import *
from sympy.abc import x, y, rho, phi, v, theta


def model2(state, action):
    'model for motion'
    n, u = len(state), len(action)
    state_dot = np.zeros(n)
    speed, car_length = 1, 1
    state_dot[0] = action[0] * np.cos(state[2]) * speed
    state_dot[1] = action[0] * np.sin(state[2]) * speed
    state_dot[2] = np.tan(action[1]) * speed / car_length
    return state_dot


def get_action_list():
    action_list = []
    max_speed = 40
    max_angle = 0.4
    angle = - max_angle
    speed = - max_speed
    num_of_actions = 20
    while speed <= max_speed:
        while angle <= max_angle:
            action_list.append([speed, angle])
            angle += 2 * max_angle / num_of_actions
        speed += 2 * max_speed / num_of_actions
        angle = - max_angle
    return action_list


model = Matrix([v * cos(theta), v * sin(theta), tan(phi)])
state_variables = Matrix([x, y, theta])
input_variables = Matrix([v, phi])
init_x = 56.34
init_y = 23.48
init_theta = 0.12
actions = get_action_list()
jac_1 = model.jacobian(state_variables)
jac_2 = model.jacobian(input_variables)
print(jac_1), print(jac_2)
for action in actions:
    values = [(x, init_x), (y, init_y), (theta, init_theta), (v, action[0]), (phi, action[1])]
    p = (jac_1*state_variables).subs(values)
    q = (jac_2*input_variables).subs(values)
    print(action, p+q, model2([init_x, init_y, init_theta], action))
