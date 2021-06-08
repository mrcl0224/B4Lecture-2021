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

    if dim == 2:

        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel="x", ylabel="y")
        ax.set_title(filename)
        plt.scatter(dt_np[0], dt_np[1])
        plt.show()

    elif dim == 3:

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.set_title(filename)
        ax.plot(dt_np[0], dt_np[1], dt_np[2], marker="o", linestyle="None")
        plt.show()

    else:
        print("Warning! : dim must be 2 or 3.")


def eigen_v(dt_np, nml=False):

    diff_dt = dt_np - np.mean(dt_np)

    # print(diff_dt)

    if nml == True:
        for i in range(dt_np.shape[1]):
            diff_dt[:, i] /= np.std(dt_np[:, i])

    conv_dt = (diff_dt.T @ diff_dt) / dt_np.shape[0]

    eig_vec = LA.eig(conv_dt)[1]

    cont_rate = LA.eig(conv_dt)[0] / np.sum(LA.eig(conv_dt)[0])

    return eig_vec, cont_rate


if __name__ == "__main__":

    args = sys.argv

    dt = pd.read_csv(args[1], header=0)

    dt_np = dt.to_numpy()

    dim = len(dt.columns)

    plot_csv(dt_np.T, dim, args[1])

    eig_vec, cont_rate = eigen_v(dt_np)
    eig_vec_nml, cont_rate_nml = eigen_v(dt_np, nml=True)

    print(eig_vec)
    print(cont_rate)

    # print(eig_vec_nml)
    # print(cont_rate_nml)

    if dim == 2:
        x = np.linspace(np.min(dt_np.T[0]), np.max(dt_np.T[0]), 1000)
        y = x * (eig_vec[0][0] / (-eig_vec[0][1]))
        y2 = x * (eig_vec[1][0] / (-eig_vec[1][1]))

        label_1 = "contribution rate = " + str(cont_rate[0])
        label_2 = "contribution rate = " + str(cont_rate[1])

        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel="x", ylabel="y")
        plt.scatter(dt_np.T[0], dt_np.T[1], label="data")
        plt.plot(x, y, color="#FF0000", label=label_1)
        plt.plot(x, y2, color="#FF00FF", label=label_2)
        plt.legend(
            bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=1, fontsize=10
        )
        plt.show()

    if dim == 3:

        a = np.arange(-1.5, 1.5, 0.01)

        x = np.linspace(np.min(dt_np.T[0]), np.max(dt_np.T[0]), 1000)
        y = x * (eig_vec[0][0] / (eig_vec[1][0]))
        z = x * (eig_vec[0][0] / (eig_vec[2][0]))

        y2 = x * (eig_vec[0][1] / (eig_vec[1][1]))
        z2 = x * (eig_vec[0][1] / (eig_vec[2][1]))

        y3 = x * (eig_vec[0][2] / (eig_vec[1][2]))
        z3 = x * (eig_vec[0][2] / (eig_vec[2][2]))

        label_1 = "contribution rate = " + str(cont_rate[0])
        label_2 = "contribution rate = " + str(cont_rate[1])
        label_3 = "contribution rate = " + str(cont_rate[2])

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.plot(
            dt_np.T[0],
            dt_np.T[1],
            dt_np.T[2],
            marker="o",
            linestyle="None",
            label="data",
        )
        ax.plot(
            x, y, z, marker="o", ms=2, color="#FF0000", linestyle="None", label=label_1
        )
        ax.plot(
            x,
            y2,
            z2,
            marker="o",
            ms=2,
            color="#FF00FF",
            linestyle="None",
            label=label_2,
        )
        ax.plot(
            x,
            y3,
            z3,
            marker="o",
            ms=2,
            color="#00FF00",
            linestyle="None",
            label=label_3,
        )
        plt.legend(loc="upper left", fontsize=10)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel="z1", ylabel="z2")

        eig_z1 = (
            eig_vec[0][0] * dt_np.T[0]
            + eig_vec[1][0] * dt_np.T[1]
            + eig_vec[2][0] * dt_np.T[2]
        )
        eig_z2 = (
            eig_vec[0][1] * dt_np.T[0]
            + eig_vec[1][1] * dt_np.T[1]
            + eig_vec[2][1] * dt_np.T[2]
        )

        plt.scatter(eig_z1, eig_z2, color="#0000FF", label="PCA")

        eig_z1_nml = (
            eig_vec_nml[0][0] * dt_np.T[0]
            + eig_vec_nml[1][0] * dt_np.T[1]
            + eig_vec_nml[2][0] * dt_np.T[2]
        )
        eig_z2_nml = (
            eig_vec_nml[0][1] * dt_np.T[0]
            + eig_vec_nml[1][1] * dt_np.T[1]
            + eig_vec_nml[2][1] * dt_np.T[2]
        )

        plt.scatter(eig_z1_nml, eig_z2_nml, color="#FF0000", label="PCA with nml")
        plt.legend(
            bbox_to_anchor=(1, 1), loc="upper right", borderaxespad=1, fontsize=10
        )
        plt.show()

    if dim >= 4:

        fig = plt.figure()
        ax = fig.add_subplot(
            111, xlabel="Principal component", ylabel="cumulative contribution ratio"
        )
        ax.plot(np.cumsum(cont_rate), color="#FF0000", label="PCA")
        ax.plot(np.cumsum(cont_rate_nml), color="#0000FF", label="PCA with nomalize")
        p = plt.plot([0, 100], [0.9, 0.9], "#000000", linestyle="dashed")
        plt.legend(
            bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=1, fontsize=10
        )
        plt.show()

        sum_rate = 0
        k = 0

        while 1:

            sum_rate += cont_rate[k]

            if sum_rate > 0.90:

                print(
                    "dimension : ",
                    k + 1,
                    "\n",
                    "cumulative contribution ratio : ",
                    sum_rate,
                )
                break

            k += 1

        sum_rate = 0
        k = 0

        while 1:

            sum_rate += cont_rate_nml[k]

            if sum_rate > 0.90:

                print(
                    "dimension : ",
                    k + 1,
                    "\n",
                    "cumulative contribution ratio : ",
                    sum_rate,
                )
                break

            k += 1
