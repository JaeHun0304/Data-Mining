import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


X = np.array([[9, 22], [0, 2], [8, 19], [10, 18], [1, 2]])

# Compute mean by using numpy mean(array) function and print mean vector
sample_mean = np.mean(X.T, axis=1)
print("Sample mean vector of D: \n" + str(sample_mean))

# original data matrix D
print("Data matrix D: \n" + str(X))

# Print N normalized covariance matrix
cov_X = np.cov(X.T)
print("N normalized Covariance Matrix: \n" + str(cov_X.T))

# Compute eigenvalues and eigenvector of covariance
cov_X_eig_val, cov_X_eig_vec = np.linalg.eig(cov_X)
print("Eigenvalues of Covariance Matrix: \n" + str(cov_X_eig_val))
print("Eigenvector of Covariance Matrix: \n" + str(cov_X_eig_vec))

# First principal component will be the attribute which has higher eigenvalue
# In this case, X2 in D matrix
First_Principal = np.sort(cov_X_eig_val)
print("First principal component of D: " + str(First_Principal[-1]))

# By doing dot product with transpose of eigenvector and transpose of original D
# matrix, we can get projected coordinate of first principal component
Projected_X = cov_X_eig_vec.T.dot(X.T)
print("Projected coordinate of D: \n" + str(Projected_X))


tot = sum(cov_X_eig_val) # total sum of eigenvalues
var_exp = [(i / tot)*100 for i in sorted(cov_X_eig_val, reverse=True)] 
# Divide each eigenvalue with total sum of them
print(var_exp)
