import numpy as np

xs = np.array([(0.5,4.5), (2.2,1.5), (3.9,3.5), (2.1,1.9), (0.5,3.2), (0.8,4.3), 
               (2.7,1.1), (2.5,3.5), (2.8,3.9), (0.1,4.1)])
thetas = np.array([[1/3, 2/3], [1/3, 2/3], [1/3, 2/3]])


ws_A = []
ws_B = []
ws_C = []

vs_A = []
vs_B = []
vs_C = []

ll_new = 0

# E-step: calculate probability distributions over possible completions
for x in xs:


    ll_A = np.sum([x*np.log(thetas[0])])
    ll_B = np.sum([x*np.log(thetas[1])])
    ll_C = np.sum([x*np.log(thetas[2])])

    # [EQN 1]
    denom = np.exp(ll_A) + np.exp(ll_B) + np.exp(ll_C)
    w_A = np.exp(ll_A)/denom
    w_B = np.exp(ll_B)/denom
    w_C = np.exp(ll_C)/denom 

    ws_A.append(w_A)
    ws_B.append(w_B)
    ws_C.append(w_C)

    # used for calculating theta
    vs_A.append(np.dot(w_A, x))
    vs_B.append(np.dot(w_B, x))
    vs_C.append(np.dot(w_C, x))

    # update complete log likelihood
    ll_new += w_A * ll_A + w_B * ll_B + w_C * ll_C

# M-step: update values for parameters given current distribution
# [EQN 2]
thetas[0] = np.sum(vs_A, 0)/np.sum(vs_A)
thetas[1] = np.sum(vs_B, 0)/np.sum(vs_B)
thetas[2] = np.sum(vs_C, 0)/np.sum(vs_C)
# print distribution of z for each x and current parameter estimate

print ("Iteration: 1")
print ("theta_A = {} theta_B = {}, theta_C = {} new_loglikeihood = {}".format(thetas[0], thetas[1], thetas[2], ll_new))