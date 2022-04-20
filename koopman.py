'Implementation of Koopman operator with Fourier feature and testing it with acasxu data samples'

import os
import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

np.set_printoptions(threshold=np.inf)


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


def car_model_simulation(init_state, actions, steps, h):
    """Simulating the car model for given state and control inputs"""
    state = init_state
    states = np.empty((4, 0), float)

    states = np.hstack((states, state.reshape(4, 1)))
    for i in range(steps):
        state = state + h * model(state, actions[:, i])
        states = np.hstack((states, state.reshape(4, 1)))
    return states


def generate_simulation_states(num_sims):
    """Generates states through actual simulation of the system.
      Output is a list of matrices where each matrix contains
      all the states captured during a full simulation.
      Each matrix contains all x, y, theta, cmd generated during
      steps of the simulation"""

    # parameter
    t_final = 1
    steps = 30

    # initial set
    x_delta = 0.00
    y_delta = 0.00
    phi_delta = 0.00
    theta_delta = 0.0

    v_mid = 3
    v_delta = 0.1

    # acceleration inputs
    a_mid = 5
    a_delta = 2

    # steering angle inputs
    s_mid = 0.0
    s_delta = 0.20

    # generate the training set by simulating the vehicle model
    X = []
    U = []

    for i in range(0, num_sims):
        # construct random initial state
        x = np.random.uniform(low=-x_delta, high=x_delta)
        y = np.random.uniform(low=-y_delta, high=y_delta)
        phi = np.random.uniform(low=-phi_delta, high=phi_delta)
        theta = np.random.uniform(low=-theta_delta, high=theta_delta)

        x0 = np.array([x, y, phi, theta])

        # construct random control inputs
        u_acc = np.random.uniform(low=a_mid - a_delta, high=a_mid + a_delta, size=(1, steps))
        u_steer = np.random.uniform(low=s_mid - s_delta, high=s_mid + s_delta, size=(1, steps))

        u = np.concatenate((u_acc, u_steer), axis=0)

        # simulate vehicle model
        tmp = car_model_simulation(x0, u, steps, 0.01)

        X.append(tmp)
        U.append(u)

    return X, U


def generate_Fourier(n, count):
    """Generating a set of Wi and bi for Fourier features
     to be used in obsevables g()"""

    lw = []
    lb = []
    l = 1
    np.random.seed(0)
    for i in range(count):
        WT = stats.norm.rvs(loc=0, scale=l, size=n)
        b = stats.uniform.rvs(loc=0, scale=2 * np.pi, size=1)
        lw.append(WT)
        lb.append(b)
    return lw, lb


def g(X, WT, b):
    """creating observables g1(x), g2(x), ..., gn(x).
     We generate them using Fourier feature
     g(X) = cos(wT*X + b)"""
    out = np.cos(np.dot(WT, X) + b)
    return out


def DMD(X, U, rank):
    'Dynamic Mode Decomposition'

    tmp = X[0]
    X1 = tmp[:, 0:tmp.shape[1] - 1]
    X2 = tmp[:, 1:tmp.shape[1]]
    for i in range(1, len(X)):
        tmp = np.array(X[i])
        X1 = np.concatenate((X1, tmp[:, 0:tmp.shape[1] - 1]), axis=1)
        X2 = np.concatenate((X2, tmp[:, 1:tmp.shape[1]]), axis=1)

    U_ = np.array(U[0])

    for i in range(1, len(U)):
        U_ = np.concatenate((U_, np.array(U[i])), axis=1)

    X1 = np.concatenate((X1, U_), axis=0)

    # singular value decomposition

    V, S, W = np.linalg.svd(X1, full_matrices=False)
    # reduce rank by removing the smallest singular values
    # print(V, S, W)
    V = V[:, 0:rank]
    S = S[0:rank]
    W = W[0:rank, :]

    AA = np.linalg.multi_dot((X2, np.transpose(W), np.diag(np.divide(1, S)), np.transpose(V)))

    # divide into state matrix and input matrix

    B = AA[:, AA.shape[0]: AA.shape[1]]
    A = AA[:, 0: AA.shape[0]]
    print("Shapes", A.shape, B.shape)

    return A, B


def generate_xp(s, WF, BF):
    out = np.zeros((len(WF), s.shape[1]))
    for i in range(len(WF)):  # iterating thorough each wi
        for r in range(s.shape[1]):
            out[i, r] = g(s[:, r], WF[i], BF[i])
    res = np.concatenate((s, out))
    return res


def predict(xs, us, A, B, num_observables, WF, BF):
    g_xs = generate_xp(xs, WF, BF)
    us = us.reshape(us.shape[0], 1)
    out = np.dot(A, g_xs) + np.dot(B, us)
    return out


def preprocessing(sim_states, sim_cmds):
    'Normalizing and scaling the dataset'

    'concatenating all arrays in sim_states column-wise to get the norm'
    tmp_s = np.zeros((sim_states[0].shape[0], 0))
    for a in sim_states:
        for j in range(a.shape[1]):
            x = np.transpose(np.array([a[:, j]]))
            tmp_s = np.concatenate((tmp_s, x), axis=1)
    tmp_c = np.zeros((sim_cmds[0].shape[0], 0))
    for a in sim_cmds:
        for j in range(a.shape[1]):
            x = np.transpose(np.array([a[:, j]]))
            tmp_c = np.concatenate((tmp_c, x), axis=1)
    print("tmp_c", tmp_c.shape)

    norm_s_x = np.linalg.norm(tmp_s[0, :])
    norm_s_x_dot = np.linalg.norm(tmp_s[1, :])
    norm_s_phi = np.linalg.norm(tmp_s[2, :])
    norm_s_theta = np.linalg.norm(tmp_s[3, :])


    norm_u_acc = np.linalg.norm(tmp_c[0, :])
    norm_u_steer = np.linalg.norm(tmp_c[1, :])



    normalized_sim_states = []

    for a in sim_states:
        a[0, :] = a[0, :] / norm_s_x
        a[1, :] = a[1, :] / norm_s_x_dot
        a[2, :] = a[2, :] / norm_s_phi
        a[3, :] = a[2, :] / norm_s_theta

    for a in sim_cmds:
        a[0, :] = a[0, :] / norm_u_acc
        a[1, :] = a[1, :] / norm_u_steer

    norms = [norm_s_x, norm_s_x_dot, norm_s_theta, norm_s_phi, norm_u_acc, norm_u_steer]
    print("norms", norms)
    return sim_states, sim_cmds, norms


def Test(init_test_state, test_cmds, A, B, num_observables, WF, BF):
    s = init_test_state[:, 0]
    cmds = test_cmds
    outx = [s]
    for i in range(cmds.shape[1]):
        tmp = np.transpose(np.array([s]))
        p = predict(tmp, cmds[:, i], A, B, num_observables, WF, BF)
        s = p[0:4, 0]
        # print(s, init_test_state[:, i + 1])
        outx.append(s)

    'plot'

    outx = np.array(outx)
    # print(outx.shape)
    # print(outx, init_test_state[0,:])
    rms1 = mean_squared_error(np.array(outx)[:,0], init_test_state[0, :])
    rms2 = mean_squared_error(np.array(outx)[:,1], init_test_state[1, :])

    print("ROOT MEAN SQUARE ERROR: ", rms1, rms2)
    # fig,ax = plt.subplots(3)
    # ax[0].plot(outx[:, 0], '--', color="r")
    # ax[0].plot(init_test_state[0, :], '.', color="b")
    # ax[1].plot(outx[:, 1], '--', color="r")
    # ax[1].plot(init_test_state[1, :], '.', color="b")
    plt.plot(outx[:, 0], outx[:, 1], '-', color="r")
    plt.plot(init_test_state[0, :],init_test_state[1, :], '-', color="b")
    # ax[2].plot(outx[:, 2], '--', color="r")
    # ax[2].plot(init_test_state[2, :], '.', color="b")
    # plt.xlabel("number of samples")
    # plt.ylabel("Acasxu intruder x position")
    # plt.legend(["Predicted x positions", "Simulated x positions"])
    # plt.show()

def main():
    sim_nums = 500

    sim_states, sim_cmds = generate_simulation_states(sim_nums)
    print(sim_states[0].shape, sim_cmds[0].shape)
    if len(sim_states) == 0:
        print("Empty training set.")
        return

    n_sim_states, n_sim_cmds, _ = preprocessing(sim_states, sim_cmds)

    rate = 0.9  # rate of train data
    train_size = int(np.round(sim_nums * rate))  # train dataset size
    test_states = n_sim_states[train_size:]  # test
    test_cmds = n_sim_cmds[train_size:]  # test
    states = n_sim_states[0: train_size]  # train
    cmds = n_sim_cmds[0: train_size]  # train

    'Generate Fourier matrices for count number of observables'
    num_observables = 8
    rank = num_observables
    X = [0, 0, 0, 0]
    XP = []
    WF, BF = generate_Fourier(len(X), num_observables)

    'making input from states and observables for DMD'
    for s in states:
        xps = generate_xp(s, WF, BF)
        XP.append(xps)
    print("DMD")
    A, B = DMD(XP, cmds, rank)  # Koopman operators

    for i in range(10):
        Test(states[i], cmds[i], A, B, num_observables, WF, BF)
    plt.show()

if __name__ == "__main__":
    main()
