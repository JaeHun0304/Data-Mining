
import matplotlib.pyplot as plt
import numpy as np

# define x-axis with linespace from 0 to 18.0 with 5000 samples
# and y with 2D list that each has 5000 zero element array
x = np.linspace(0.0, 18.0, num=5000)
y = np.array([np.zeros(5000)] * 5)

# define samples
samples = [1, 5, 6, 9, 15]

# if the windows falls into 3/2 = 1.5, accumulate 1/15(1/nh), otherwise accumulate zero
for i in range(len(samples)):
	for j in range(len(x)):
		if abs(x[j] - samples[i]) <= 1.5:
			y[i][j] += (1/15)
		else:
			y[i][j] += 0

# print all piecewise constant functions
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

# print total convoluted function
y_total = y[0] + y[1] + y[2] + y[3] + y[4]
plt.plot(x, y_total)
plt.show()