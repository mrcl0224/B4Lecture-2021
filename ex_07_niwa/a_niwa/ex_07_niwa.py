import sys
import math
import numpy as np
import numpy.linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy as sp


def plot_csv(dt_np, dim, filename):

    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel="x", ylabel="y")
    ax.set_title(filename)

    colortype = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", "#F00000", "#00F000", "#0000F0", "#F0F000", "#00F0F0", "#F000F0"]

    for i in range(dim):

        plt.plot(dt_np[i], color = colortype[i])

    plt.show()

def minimax(dt_np, dim, k):

    init_vec = dt_np[random.randrange(dt_np.shape[0])]

    tmp = dt_np - init_vec

    dt_dist = np.linalg.norm(tmp)

    far_vec = dt_np[np.argmax(dt_dist)]

    cent_vec = 0.5 * (init_vec + far_vec)

    s = np.zeros((k, dt_np.shape[0], dim))

    if dim == 1:

        count = np.zeros(k)

        for i in range(dt_np.shape[0]):

            if dt_np[i] < cent_vec:

                s[0, i] += dt_np[i]
                count[0] += 1

            elif dt_np[i] >= cent_vec:

                s[1, i] += dt_np[i]
                count[1] += 1

    #2次元だけとりあえず想定
    if dim > 1:

        slope = (far_vec[1] - init_vec[1]) / (far_vec[0] - init_vec[0])

        std_line = slope * (dt_np[:, 0] -  cent_vec[0]) + cent_vec[1]

        count = np.zeros(k)

        for i in range(dt_np.shape[0]):

            if dt_np[i, 1] < std_line[i]:

                s[0, i] += dt_np[i]
                count[0] += 1

            elif dt_np[i, 1] >= std_line[i]:

                s[1, i] += dt_np[i]
                count[1] += 1

    return s, count

def gaussian(x, mean_x, sd_x, dim):

    #print("mean_x : ", mean_x)
    #print("sd_x : ", sd_x)
    #print("1 / (pow(2 * math.pi, dim/2) : ", 1 / (pow(2 * math.pi, dim/2)))
    #print("pow(sd_x, 1/2) : ", pow(sd_x, 1/2))
    #print("np.linalg.inv(sd_x) : ", np.linalg.inv(sd_x).shape)
    #print("math.exp((-(x - mean_x).T @ np.linalg.inv(sd_x) @ (x - mean_x))/2) : ", np.exp((-(x - mean_x) @ np.linalg.inv(sd_x) @ (x - mean_x).T)/2))
    #print("x - mean_x : ", (x - mean_x).shape)
    #print("(x - mean_x).T : ", ((x - mean_x).T).shape)

    n = np.zeros(x.shape[0])

    for i in range(x.shape[0]):
        n = (1 / (pow(2 * math.pi, dim/2) * pow(np.linalg.norm(sd_x, ord=2), 1/2))) * np.exp((-(x[i] - mean_x) @ np.linalg.inv(sd_x) @ (x[i] - mean_x).T)/2)
    #print("gaussian : " , n)

    return n

def em_algorithm(x, mean_x, sd_x, k, dim, pi):

    #E step

    gamma = np.zeros([k, x.shape[0]])

    x_gauss = np.zeros([k, x.shape[0], x.shape[1]])

    for i in range(k):

        x_gauss[i] +=  gaussian(x, mean_x[i], sd_x[i], dim)

    #print(x_gauss)
    print(pi)

    print(np.sum(pi * x_gauss))

    for i in range(k):

        gamma[i] += (pi[i] * x_gauss[i]) / (np.sum(pi * x_gauss))

    print("gamma : ", gamma.shape)

    n_k = np.sum(gamma, axis = 1)
    print("n_k : ", n_k)

    #M step

    mean_x_new = np.zeros(k)
    sd_x_new = np.zeros([k, dim, dim])

    for i in range(k):

        for i in range(x.shape[0]):

            mean_x_new = 1/n_k * np.sum(gamma * x)

            sd_x_new = 1/n_k * np.sum(gamma * x[i]) * (x[i] - mean_x_new) @ (x[i] - mean_x_new).T

        pi_new = n_k / x.shape[0]

    return mean_x_new, sd_x_new, pi_new

def likelihood(x, mean_x, sd_x, pi, dim, k):

    tmp = np.zeros(x.shape[0])

    for i in range(k):

        for j in range(x.shape[0]):

            tmp[j] += pi[i] * (1 / (pow(math.pi, dim/2) * pow(np.linalg.norm(sd_x[i], ord=2), 1/2))) * np.exp((-(x[j] - mean_x[i]) @ np.linalg.inv(sd_x[i]) @ (x[j] - mean_x[i]).T)/2)

    n = np.sum(np.log(tmp))

    return n

if __name__ == "__main__":

    args = sys.argv

    dt = pd.read_csv(args[1], header=0)

    dt_np = dt.to_numpy()

    k = 2

    dim = len(dt.columns)

    plot_csv(dt_np.T, dim, args[1])

    s, count = minimax(dt_np, dim, k)

    pi_k = count / dt_np.shape[0]

    mean_x = np.zeros(k)
    sd_x = np.zeros([k, dim, dim])

    #print(s[0])

    #print((s[0])[~np.any(s[0] == 0, axis = 1)])
    #print((s[1])[~np.any(s[1] == 0, axis = 1)])

    for i in range(k):

        mean_x[i] += np.mean((s[i])[~np.any(s[i] == 0, axis = 1)])

        sd_x[i] += np.cov((s[i])[~np.any(s[i] == 0, axis = 1)], rowvar = False, bias = True)

        #print(sd_x[i].shape)

    #print("mean_x : ", mean_x)
    #print("sd_x : ", sd_x)

    L_old = 0

    L_new = 1

    eps = 0.001

    while(L_new - L_old > 0.01):

        mean_x, sd_x, pi_k = em_algorithm(dt_np, mean_x, sd_x, k, dim, pi_k)

        L_old = L_new

        print("sd_x : ", sd_x.shape)

        L_new = likelihood(dt_np, mean_x, sd_x, pi_k, dim, k)

    print("end! L_new - L_old = ", L_new - L_old, "< 0.01")

    x = [k, np.arange(-3, 3, 0.01)]

    gauss = np.zeros_like(x)

    for i in range(k):

        gauss[i] += gaussian(x, mean_x, sd_x, dim)

    if dim == 1:

        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel="x", ylabel="y")
        ax.set_title(filename)

        plt.scatter(dt_np, 0)

        colortype = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", "#F00000", "#00F000", "#0000F0", "#F0F000", "#00F0F0", "#F000F0"]

        for i in range(dim):

            plt.plot(x[i], gauss[i], color = colortype[i])

        plt.show()

    elif dim == 2:

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.set_title(filename)
        ax.plot(dt_np[0], dt_np[1], dt_np[2], marker="o", linestyle="None")
        plt.show()
