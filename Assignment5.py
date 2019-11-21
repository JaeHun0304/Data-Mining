import numpy as np
import scipy.stats
import math



X1 = np.array([5,7,3,6])
X2 = np.array([8,7,4,5,1])


print(scipy.stats.norm(np.mean(X1), np.std(X1)).pdf(1.0) * float(4/9))
print(scipy.stats.norm(np.mean(X2), np.std(X2)).pdf(1.0) * float(5/9))


X1_Mean = np.array([1., 3.])
X1_Cov = np.array([[5.,3.], [3.,2.]])
X2_Mean = np.array([5.,5.])
X2_Cov = np.array([[2.,0.], [0.,1.]])

print(scipy.stats.multivariate_normal(mean=X1_Mean, cov=X1_Cov).pdf([3, 4]) * 0.5)
print(scipy.stats.multivariate_normal(mean=X2_Mean, cov=X2_Cov).pdf([3, 4]) * 0.5)

feature = np.array([[1, 1], [2, 2], [1, 1], [3, 2], [1, 2], [3, 2]])

# All possible combination of dom(X)
v = [[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]
dyl = [0 for _ in range(len(v))]
dyh = [0 for _ in range(len(v))]
dnl = [0 for _ in range(len(v))]
dnh = [0 for _ in range(len(v))]

# find Nvi by loop
for i in range(len(v)):
    for j in range(len(feature)):
        if feature[j][0] in v[i] and feature[j][1] == 1:
            dyl[i] = dyl[i] + 1
        elif feature[j][0] in v[i] and feature[j][1] == 2:
            dyh[i] = dyh[i] + 1
        elif feature[j][0] not in v[i] and feature[j][1] == 1:
            dnl[i] = dnl[i] + 1
        else:
            dnh[i] = dnh[i] + 1

# calculate probabilityof Dy and Dh        
p_dyl = [0 for _ in range(len(v))]
p_dyh = [0 for _ in range(len(v))]
p_dnl = [0 for _ in range(len(v))]
p_dnh = [0 for _ in range(len(v))]
for i in range(len(dyl)):
    if dyl[i]+dyh[i] != 0:
        p_dyl[i] = float(dyl[i]/(dyl[i]+dyh[i]))
    p_dyh[i] = 1 - p_dyl[i]
    if dnl[i]+dnh[i] != 0:
        p_dnl[i] = float(dnl[i]/(dnl[i]+dnh[i]))
    p_dnh[i] = 1 - p_dnl[i]
    
h_dy = [0 for _ in range(len(v))]
h_dn = [0 for _ in range(len(v))]

# calculate entropy based on the probability of Dy and Dh for each class
for i in range(len(dyl)):
    if p_dyl[i] != 0 and p_dyh[i] != 0:
        h_dy[i] = -(p_dyl[i] * math.log2(p_dyl[i]) + p_dyh[i] * math.log2(p_dyh[i]))
    if p_dnl[i] != 0 and p_dnh[i] != 0:
        h_dn[i] = -(p_dnl[i] * math.log2(p_dnl[i]) + p_dnh[i] * math.log2(p_dnh[i]))

h_dydn = [0 for _ in range(len(v))]
gain = [0 for _ in range(len(v))]

# calculate gain for all possible cases and print results
for i in range(len(dyl)):
    h_dydn[i] = (float((dyl[i] + dyh[i])/(dyl[i]+dyh[i]+dnl[i]+dnh[i])) * h_dy[i]) + (float((dnl[i] + dnh[i])/(dyl[i]+dyh[i]+dnl[i]+dnh[i])) * h_dn[i])
    gain[i] = (-(float(2/3) * math.log2(float(2/3)) + float(1/3) * math.log2(float(1/3)))) - h_dydn[i]

print(gain)
print(max(gain))