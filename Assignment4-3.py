
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.0, 18.0, num=5000)
y = np.array([np.zeros(5000)] * 5)

samples = [1, 5, 6, 9, 15]

for i in range(len(samples)):
	for j in range(len(x)):
		if abs(x[j] - samples[i]) <= 1.5:
			y[i][j] += (1/15)
		else:
			y[i][j] += 0

plt.plot(x, y[0])
plt.show()
plt.plot(x, y[1])
plt.show()
plt.plot(x, y[2])
plt.show()
plt.plot(x, y[3])
plt.show()
plt.plot(x, y[4])
plt.show()