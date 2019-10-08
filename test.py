import numpy as np

X = np.array([[6, 1, 2, 4, -3, -4, -3], [11, 1, 3, 6, -2, -1, -5], [3, -1, 5, 2, 0, 2, 2]])

X1 = [6, 1, 2, 4, -3, -4, -3]
X2 = [11, 1, 3, 6, -2, -1, -5]
X3 = [3, -1, 5, 2, 0, 2, 2]
sample_mean = np.mean(X)
print(sample_mean)

cov_X = np.cov(X)
print(cov_X)

cov_X_eig_val, cov_X_eig_vec = np.linalg.eig(cov_X)
np.corrcoef
print(cov_X_eig_val)
print(cov_X_eig_vec)

Projected_X = cov_X_eig_vec.T.dot(X)
print("Projected coordinate of D: \n" + str(Projected_X))

print(np.corrcoef([X1, X2]))
print(np.corrcoef([X2, X3]))
print(np.corrcoef([X1, X3]))