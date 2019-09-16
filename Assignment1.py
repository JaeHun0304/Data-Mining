import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Data age and weight as python lists
X = [69, 74, 68, 70, 72, 67, 66, 70, 76, 68, 72, 79, 74, 67, 66, 71, 74, 75, 75, 76]
Y = [153, 175, 155, 135, 172, 150, 115, 137, 200, 130, 140, 265, 185, 112, 140, 150, 165, 185, 210, 220]

# Mean can be derived by (sum) / (number of elements)
mean_x = sum(X)/len(X)
print("Sample mean X: " + str(mean_x))

# Python stats module mode function returns highest frequency element
mode_x = stats.mode(X)
print(mode_x)

# numpy median function is called
median_x = np.median(X)
print("Median of X: " + str(median_x))

# sample mean of Y is also derived like sample mean of X
mean_y = sum(Y)/len(Y)
print("Sample mean Y: " + str(mean_y))

# numpy sample variance function is called with n and (n-1) denominator which give 
# biased and unbiased sample variance
sample_var_y = np.var(Y)
sample_var_y_unbiased = np.var(Y, ddof=1)
print("Sample variance Y: " + str(sample_var_y))
print("Unbiased sample variance Y: " + str(sample_var_y_unbiased))

# create line space based on the sigma and mu value of the data X
# and plot normal distribution PDF


# find frequency in X which X > 80
x_freq_80 = 0
for i in range(len(X)):
	if X[i] > 80: 
		x_freq_80 += 1
print("Frequency of X > 80: " + str(x_freq_80))

# 2-D mean is just tuple of sample mean of X and Y
d2_mean = (mean_x, mean_y)
print("2D-mean: " + str(d2_mean))

# Covariance matrix generated with N normalized bias True
covariance_matrix = np.cov([X,Y], bias=True)
print("N normalized Covariacne Matrix: " + str(covariance_matrix))

# np.corrcoef generates all cases as 2-D array which are [[XX, XY], [YX, YY]}
# I used [XY] case (Same with [YX] value)
correlation = np.corrcoef([X,Y])[1,0]
print("Correaltion of X and Y: " + str(correlation))


sigma = np.std(X, ddof=1)
print(sigma)
mu = mean_x
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.show()

# Scatter plot created by Python matplotlib module
plt.scatter(X, Y)
plt.title('Scatter plot between age and weight')
plt.xlabel('age')
plt.ylabel('weight')
plt.show()
