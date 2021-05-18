import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def regression(x1, x2, len, dim):
    x1_calc = np.zeros((len, dim + 1))

    lamb = 1

    for i in range(len):
        for j in range(dim + 1):
            x1_calc[i][j] += np.power(x1[i], j)
    # 正規化しないver
    # ans = np.linalg.inv(x1_calc.T@x1_calc) @ x1_calc.T @ x2
    ans = (
        np.linalg.inv(x1_calc.T @ x1_calc + (lamb * np.identity(dim + 1)))
        @ x1_calc.T
        @ x2
    )

    return ans


def generate_y(x, a, dim):

    y = 0

    for i in range(dim + 1):
        y += a[dim - i] * np.power(x, dim - i)

    return y


if __name__ == "__main__":
    # data1.csv
    dt1 = pd.read_csv("data1.csv", header=0)
    # print(dt1)
    dim_1 = 3
    x1 = np.array(dt1["x1"]).T
    x2 = np.array(dt1["x2"])
    # print(x1.shape)
    a = regression(x1, x2, len(dt1), dim_1)
    print(a)
    # print(b)
    x = np.linspace(np.min(x1) - 0.5, np.max(x1) + 0.5, 100)
    # y = (5/2)*x
    y = generate_y(x, a, dim_1)

    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel="x1", ylabel="x2")
    ax.plot(x, y, color="#FF0000")
    plt.scatter(dt1["x1"], dt1["x2"])
    plt.show()

    # data2.csv
    dt2 = pd.read_csv("data2.csv", header=0)

    dim_2 = 3

    x1 = np.array(dt2["x1"]).T
    x2 = np.array(dt2["x2"])
    # print(x1.shape)
    a = regression(x1, x2, len(dt2), dim_2)
    print(a)
    # print(b)
    x = np.linspace(np.min(x1) - 0.5, np.max(x1) + 0.5, 100)
    # y = (5/2)*x
    y = generate_y(x, a, dim_2)

    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel="x1", ylabel="x2")
    ax.plot(x, y, color="#FF0000")
    plt.scatter(dt2["x1"], dt2["x2"])
    plt.show()

    # data3.csv
    dt3 = pd.read_csv("data3.csv", header=0)

    dim_3 = 3

    x1 = np.array(dt3["x1"]).T
    x2 = np.array(dt3["x2"]).T
    x3 = np.array(dt3["x3"])
    # print(x1.shape)
    a = regression(x2, x3, len(dt3), dim_3)
    # print(b)
    x = np.linspace(np.min(x1) - 0.5, np.max(x1) + 0.5, 1000)
    y = np.linspace(np.min(x2) - 0.5, np.max(x2) + 0.5, 1000)
    # y = (5/2)*x
    z = generate_y(y, a, dim_3)

    X, Y = np.meshgrid(x, y)
    Z = np.array([z] * Y.shape[0])

    # print(X)
    # print(Y)
    # print(Z)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.plot_wireframe(X, Y, Z, color="#FF0000")
    ax.plot(dt3["x1"], dt3["x2"], dt3["x3"], marker="o", linestyle="None")
    plt.show()
