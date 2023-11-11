import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

N = 512  #preferably a power of 2
x = np.linspace(0,np.pi/2,N,endpoint=False)
c1 = np.cos(x)
cr1 = np.round(c1, 7) #23 bits, 7 decimalplaces

arctan_lookup = [np.arctan(1/(2**i)) for i in range(0,int(np.log2(N))+15)]
cosmult_lookup = []

for ind, arctan in enumerate(arctan_lookup):
    cosmult_lookup.append(
        (cosmult_lookup[-1] if len(cosmult_lookup) > 0 else 1) * np.cos(arctan)
            )


def cordic(theta, verbose=False):
    x, y = 1, 0
    cur_theta = 0

    #n = 10
    #for i in range(n):
    #    tan_val = np.tan(theta/n)
    #    x,y = x-y*tan_val,x*tan_val+y
    #    #c,s = np.cos(theta/n), np.sin(theta/n)
    #    #x,y = x*c-y*s,x*s+y*c

    ##return x
    #return x * (np.cos(theta/n)**n)

    cos_multiplier = 1

    for ind, arctan in enumerate(arctan_lookup):
        diff = theta - cur_theta
        sgn_diff = -1 if diff < 0 else 1
        abs_diff = np.abs(diff)
        if abs_diff < arctan:
            if verbose:
                print('skip', ind, arctan, diff)
            continue

        if verbose:
            print(ind, cur_theta, y/x, np.tan(cur_theta))
            #print(ind, x,y, cur_theta, theta, diff, arctan)
        
        tan_val = sgn_diff / (2 ** ind)
        x,y = x-y*tan_val,x*tan_val+y
        cur_theta += sgn_diff * arctan
        cos_multiplier *= np.round(np.cos(arctan), 7)

        if verbose:
            #print(x,y, tan_val)
            pass

    #return y/x
    #return x * cosmult_lookup[-1]
    return x * cos_multiplier

# tmp = np.pi * 53/180
# l = [np.cos(tmp), cordic(tmp, verbose=True)]
# for i in l:
#     print(bin(int(np.round(i * (2**10),0))))
# #print(cosmult_lookup)
# print(arctan_lookup)

c2 = np.array([cordic(theta) for theta in x])
#plt.plot(x[:100],np.tan(x[:100]))
#plt.plot(x[:100],c2      [:100])
#plt.show()

plt.plot(x, c1)
plt.plot(x, c2)
plt.show()

plt.plot(c2 -c1)
plt.show()
