import matplotlib.pyplot as plt
import numpy as np

files = ['ovbase.txt']


for filename in files:
	data = np.loadtxt(filename)

	xdata = data[:, 2]
	ydata = data[:, 0]

	plt.plot(xdata, ydata, label="DDQN")

plt.plot([0, 1550], [900, 900], label='Solved')
plt.xlabel('Training episodes')
plt.ylabel('Testing performance')
plt.legend(loc='upper right')
plt.title('Training performance of DDQN on CarRacing-v0')
plt.ylim(-150, 1000)
plt.show()