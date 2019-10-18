import matplotlib.pyplot as plt
import numpy as np


X = np.array([0.5,2.2,3.9,2.1,0.5,0.8,2.7,2.5,2.8,0.1])
Y = np.array([4.5,1.5,3.5,1.9,3.2,4.3,1.1,3.5,3.9,4.1])
means = np.array([[0.5, 4.5], [2.2,1.6], [3,3.5]])
new_means = np.array([[0.55037786, 4.04016951], [2.30305907, 1.74332438], [2.8712064,  3.44305963]])

# Plot data points(blue), initial means(yellow), 1st EM means(green)
plt.scatter(X, Y)
plt.scatter(0.5, 4.5, color="yellow")
plt.scatter(2.2, 1.6, color="yellow")
plt.scatter(3, 3.5, color="yellow")
plt.scatter(0.55037786, 4.04016951, color="green")
plt.scatter(2.30305907, 1.74332438, color="green")
plt.scatter(2.8712064,  3.44305963, color="green")
plt.title('Scatter plot of data with initial means and after 1st iter means')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()