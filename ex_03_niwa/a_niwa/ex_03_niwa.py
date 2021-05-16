import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

dt = pd.read_csv('data2.csv', header=0)
#print(dt1)

x = np.linspace(-5,5,101)
y = (5/2)*x

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='x1', ylabel='x2')
#ax.plot(x,y)
plt.scatter(dt['x1'], dt['x2'])
plt.show()
