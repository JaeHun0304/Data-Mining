import numpy as np
import matplotlib.pyplot as plt

samples = np.array([[5,8], [10,8], [11,8], [6,7], [10,7], [12,7], [13,7], [5,6], [10,6], [13,6], [6,5], [9,4], [11,5], [14,6], [15,5], 
		   [2,4], [3,4], [5,4], [6,4], [7,4], [15,4], [3,3], [7,3], [8,2]])

cluster_id = [None] * len(samples)
cores = []
cores_index = []
noise = []

def density_connected(x, k):
	for i in range(len(samples)):
		if np.linalg.norm(samples[x] - samples[i]) <= 2:
			if cluster_id[i] == None:
				cluster_id[i] = k
			if i in cores_index and cluster_id[i] == None:
				density_connected(i, k)



for i in range(len(samples)):
	counter = 0
	for j in range(len(samples)):
		if np.linalg.norm(samples[i]-samples[j]) <= 2 and i != j:
			counter = counter + 1
	if counter >= 3:
		cores.append(samples[i])
		cores_index.append(i)

print(cores)

if np.linalg.norm(samples[3] - samples[0]) <= 2:
	print("a is directly reachable from d")
else:
	print("a is not directly reachable from d")

for k in range(len(cores_index)):
	if cluster_id[cores_index[k]] == None:
		cluster_id[cores_index[k]] = k
		density_connected(cores_index[k], k)

print(cluster_id)

for i in range(len(cluster_id)):
	if cluster_id[i] == None:
		print("Sample " + str(i) + " is noise point")

for i in range(len(samples)):
	if i == 0 or i == 3 or i == 7 or i == 10:
		plt.scatter(samples[i][0], samples[i][1], color="yellow")
	if i == 1 or i == 2 or i == 4 or i == 8:
		plt.scatter(samples[i][0], samples[i][1], color="green")
	if i == 5 or i == 6 or i == 9:
		plt.scatter(samples[i][0], samples[i][1], color="blue")
	if i == 13 or i == 14:
		plt.scatter(samples[i][0], samples[i][1], color="red")
	if i == 15 or i == 16 or i ==17 or i == 21:
		plt.scatter(samples[i][0], samples[i][1], color="brown")
	if i == 18 or i == 19 or i == 22:
		plt.scatter(samples[i][0], samples[i][1], color="black")
	if i == 11 or i == 12 or i == 20 or i == 23:
		plt.scatter(samples[i][0], samples[i][1], color="purple")

plt.title('Scatter plot of density clustered data points')
plt.show()