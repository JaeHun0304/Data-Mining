from copy import deepcopy
import numpy as np

# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# Initialize data and clusters
data = np.array([[0,2], [0,0], [1.5,0], [5,0], [5,2]])
C1 = np.array([[0,2], [0,0], [5,0]])
C2 = np.array([[1.5,0], [5,2]])

#initial centroids coordinates are calculated with average of all points in the cluster
centroids = np.array([[(0+0+5)/3, (2+0+0)/3], [(1.5+5)/2, (0+2)/2]])

#print initial cluster 1, 2 and centroids
print("Initial C1:\n" + str(C1))
print("Initial C2:\n" + str(C2))
print("Initial Centroids:\n" + str(centroids))

# Initialize zero list to store old centroids in the while loop
C_old = np.zeros(centroids.shape)
# clusters information list
clusters = np.zeros(len(data))
# Error func. - Euclidean Distance between new centroids and old centroids
error = dist(centroids, C_old, None)
# Loop will run till the error becomes zero
# iteration counter for printing information at end of the loop
iteration = 1
while error != 0:
    # Assigning each points to its closest cluster
    for i in range(len(data)):
        distances = dist(data[i], centroids)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(centroids)
    # create empty list to store cluster coordinates
    cluster_coordinates = [[],[]]
    # Finding the new centroids based on the mean of the points in each cluster
    for i in range(2):
        points = [data[j] for j in range(len(data)) if clusters[j] == i]
        cluster_coordinates[i].append(points)
        centroids[i] = np.mean(points, axis=0)
    print("After " + str(iteration) + " iteration C1:\n" + str(cluster_coordinates[0]))
    print("After " + str(iteration) + " iteration C2:\n" + str(cluster_coordinates[1]))
    print("After " + str(iteration) + " iteration Centroids:\n" + str(centroids))

    # calculate errors between old and new centroids
    error = dist(centroids, C_old, None)
    print("Error of old and new centroids are:\n" + str(error))
    iteration += 1

