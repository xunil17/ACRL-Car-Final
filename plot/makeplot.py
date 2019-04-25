import matplotlib.pyplot as plt
import numpy as np

files = ['ov05.txt', 'ov08.txt', 'ov10.txt', 'ovbase.txt']


for filename in files:
	data = np.loadtxt(filename)

	xdata = data[:, 2]
	ydata = data[:, 0]

	plt.plot(xdata, ydata, label=filename)

plt.legend(loc='upper right')
plt.show()