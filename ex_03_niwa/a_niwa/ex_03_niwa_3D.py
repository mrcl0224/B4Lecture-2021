import pandas as pd
import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dt = pd.read_csv('data3.csv', header=0)
#print(dt1)
fig = plt.figure()
ax = Axes3D(fig)

ax.plot(dt['x1'],dt['x2'],dt['x3'],marker="o",linestyle='None')

plt.show()
