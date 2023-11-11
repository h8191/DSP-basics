import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def binary_round(x, r):
    if r == -1:
        return x

    mask = (1 << r)
    return np.round(x * mask, 0) / mask
#    return round(x * mask, 0) / mask
#binary_round = np.vectorize(binary_round)

cos_resolution = -1
arctan_resolution = -1
cos_resolution = 30
arctan_resolution = 30
n_iter_cordic = 30 

N = 512  #preferably a power of 2
x = np.linspace(0,np.pi/2,N,endpoint=False)

arctan_lookup = [np.arctan(1/(2**i)) for i in range(0,int(np.log2(N))+15)]

def cordic(theta, verbose=False):
    x, y = 1, 0
    cur_theta = 0

    cos_multiplier = 1

    tan_val = 2

    for iter_ind in range(n_iter_cordic):
        tan_val /= 2
        arctan = np.arctan(tan_val)
        #arctan = binary_round(arctan, arctan_resolution)

        diff = theta - cur_theta 
        #diff = binary_round(diff, arctan_resolution)

        abs_diff = np.abs(diff)
        if abs_diff < arctan:
            continue

        sgn_diff = 1 if diff > 0 else -1

        cur_theta = binary_round(cur_theta + sgn_diff * arctan, arctan_resolution)

        x,y = x-y*tan_val,x*tan_val+y
        x   = binary_round(x, cos_resolution)
        y   = binary_round(y, cos_resolution)

        cos_multiplier = binary_round(cos_multiplier * np.cos(np.arctan(tan_val)), cos_resolution)


    #return binary_round(np.cos(theta), cos_resolution)
    return binary_round(x * cos_multiplier, cos_resolution)


c1 = np.cos(x)
#c3 = binary_round(c1, 25)
c2 = np.array([cordic(theta) for theta in x])

#plt.plot(c3 -c2)
plt.figure()
plt.plot(c1 -c2)
plt.show()
