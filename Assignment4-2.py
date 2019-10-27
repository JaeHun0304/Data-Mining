from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# Initialize data points given in Figure 15.12
samples = np.array([[5,8], [10,8], [11,8], [6,7], [10,7], [12,7], [13,7], [5,6], [10,6], [13,6], [6,5], [9,4], [11,5], [14,6], [15,5], 
		   [2,4], [3,4], [5,4], [6,4], [7,4], [15,4], [3,3], [7,3], [8,2]])

# Define DBSCAN clustering with min distance 2 and min points 3 with default euclidean distance method
clustering = DBSCAN(eps=2, min_samples=3).fit(samples)
# print clustering parameters, core samples, and clustered labels
print(clustering)
print(clustering.core_sample_indices_)
print(clustering.labels_)

# Filter the samples based on the cluster id and perform scatter plot
for i in range(len(samples)):
	if clustering.labels_[i] == 0:
		plt.scatter(samples[i][0], samples[i][1], color="yellow")
	elif clustering.labels_[i] == 1:
		plt.scatter(samples[i][0], samples[i][1], color="green")

plt.title('Scatter plot of density clustered data points')
plt.show()