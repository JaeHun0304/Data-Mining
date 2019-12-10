import numpy as np
from scipy.stats import multivariate_normal

# Initialize dataset, means, mixture probabilites, and covariance matrices
xs = np.array([(0.5,4.5), (2.2,1.5), (3.9,3.5), (2.1,1.9), (0.5,3.2), (0.8,4.3), 
               (2.7,1.1), (2.5,3.5), (2.8,3.9), (0.1,4.1)])
means = np.array([[0.5, 4.5], [2.2,1.6], [3,3.5]])
prob = np.array([[1/3],[1/3],[1/3]])
cov_mat = np.array([[[1.,0.],[0.,1.]], [[1.,0.],[0.,1.]], [[1.,0.],[0.,1.]]])

# Initialzie empty weight vector
weights = [[],[],[]]

# Print initial means, mixture probabilities, and covariance matrices
print("Initial means, probabilities, and covariances:")
print("Means:")
print(means)
print("Probabilities:")
print(prob)
print("Covariances:")
print(cov_mat)

# Fill the weight vector based on the equstion in ZM book ALGORITHM 13.3
for k in range(3):
    for i in range(len(xs)):
        numer = multivariate_normal.pdf(xs[i], mean=means[k], cov=cov_mat[k]) * prob[k]
        denom = multivariate_normal.pdf(xs[i], mean=means[0], cov=cov_mat[0]) * prob[0]
        denom += multivariate_normal.pdf(xs[i], mean=means[1], cov=cov_mat[1]) * prob[1]
        denom += multivariate_normal.pdf(xs[i], mean=means[2], cov=cov_mat[2]) * prob[2]
        weights[k].append(numer/denom)

# Update means
for i in range(3):
    numer = 0
    denom = 0
    for j in range(len(xs)):
         numer += weights[i][j]*xs[j]
    denom = sum(weights[i])
    means[i] = (numer/denom)

# Update mixture probabilites
for i in range(3):
    prob[i] = (sum(weights[i])/len(xs))

# Update covariance matrices
for i in range(3):
    numer = 0
    denom = 0
    for j in range(len(xs)):
        a = xs[j]-means[i]
        numer = numer + (weights[i][j]*np.dot(a,a))
    denom = sum(weights[i])
    update = numer/denom
    cov_mat[i,0,0] = float(update)
    cov_mat[i,1,1] = float(update)

print("Means, probabilities, and covariances after 1st iteration of EM:")
print("Means:")
print(means)
print("Probabilities:")
print(prob)
print("Covariances:")
print(cov_mat)
