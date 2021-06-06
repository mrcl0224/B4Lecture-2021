import sys
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import random
from mpl_toolkits.mplot3d import Axes3D
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy as sp
import soundfile as sf

def plot_csv(dt, dim):

    if dim == 2:

        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel="x", ylabel="y")
        plt.scatter(dt["x"], dt["y"])
        plt.show()

    elif dim == 3:

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.plot(dt["x"], dt["y"], dt["z"], marker="o", linestyle="None")
        plt.show()

    else:
        print("Warning! : dim must be 2 or 3.")


def k_means(dt_np, dim, cluster, init_type):

    # 1 はランダムでの値決め
    if init_type == 1:

        tmp_cluster = cluster

        centroid = np.zeros((tmp_cluster, dim))

        for i in range(tmp_cluster):
            centroid[i] += dt_np[random.randrange(dt_np.shape[0])]

    # 2 はLBGアルゴリズム
    elif init_type == 2:

        tmp_cluster = 1

        delta = 0.1

        g = np.sum(dt_np, axis = 0) * (1/dt_np.shape[0])

        centroid = np.zeros((tmp_cluster * 2, dim))

        centroid[0] = g + numpy.full_like(g, delta)
        centroid[1] = g - numpy.full_like(g, delta)

    # 3 はミニマックス法
    elif init_type == 3:

        pass

    eps = 0.01
    prev_error = 10000000

    s = np.zeros((tmp_cluster, dt_np.shape[0], dim))
    s_dist = 0
    s_count = [0] * tmp_cluster

    while True:

        print("roop")

        s = np.zeros((tmp_cluster, dt_np.shape[0], dim))
        s_dist = 0
        s_count = [0] * tmp_cluster

        for i in range(dt_np.shape[0]):

            d = 10000
            d_tmp = 0
            fit_j = 0

            for j in range(tmp_cluster):

                d_tmp = distance(dt_np[i], centroid[j], dim)

                if d_tmp < d:
                    d = d_tmp
                    fit_j = j

            s_dist += d
            s[fit_j][i] += dt_np[i]
            s_count[fit_j] += 1

        error = s_dist / dt_np.shape[0]

        if (math.fabs(prev_error - error) / error) < eps:
            print(((prev_error - error) / error) , "< eps -> break!")
            break

        else:
            prev_error = error
            for j in range(tmp_cluster):
                centroid[j] = np.sum(s[j], axis = 0) * (1/s_count[j])

    return s

def distance(a, b, dim):

    if dim == 2:

        d = math.sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1],2))

    elif dim == 3:

        d = math.sqrt(pow(a[0] - b[0],2) + pow(a[1] - b[1],2) + pow(a[2] - b[1],2))

    return d

def pd2np(dt, dim):

    tmp = np.zeros((len(dt), dim))

    if dim == 2:

        for i in range(len(dt)):

            tmp[i] += [dt.iat[i, 0], dt.iat[i, 1]]

    elif dim == 3:

        for i in range(len(dt)):

            tmp[i] += [dt.iat[i, 0], dt.iat[i, 1], dt.iat[i, 2]]

    return tmp

def plot_result(cluster, s, dim):

    if dim == 2:

        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel="x", ylabel="y")
        colortype = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", "#F00000", "#00F000", "#0000F0", "#F0F000", "#00F0F0", "#F000F0"]
        for i in range(cluster):
            plt.scatter(s[i, :, 0], s[i, :, 1], color = colortype[i])
        plt.show()

    elif dim == 3:

            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("x3")
            colortype = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", "#F00000", "#00F000", "#0000F0", "#F0F000", "#00F0F0", "#F000F0"]
            for i in range(cluster):
                ax.plot(s[i, :, 0], s[i, :, 1], s[i, :, 2], marker="o", linestyle="None", color = colortype[i])
            plt.show()

    else:
        print("Warning! : dim must be 2 or 3.")



if __name__ == "__main__":

    args = sys.argv

    dt = pd.read_csv(args[1], header=0)

    dim = len(dt.columns)

    plot_csv(dt, dim)

    dt_np = pd2np(dt, dim)

    print(dt_np.shape[0])

    cluster = 5

    s = k_means(dt_np, dim, cluster, 1)

    plot_result(cluster, s, dim)
