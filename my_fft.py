# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 14:38:48 2023

@author: harsha pothuganti
"""


import time
import numpy as np
import matplotlib.pyplot as plt

data = np.random.randint(-1,2,32768)*64

for fft_func in [np.fft.fft,]:
    for fft_size in [1024,8192,32768]:
        t1 = time.time()
        f_abs = np.abs(fft_func(data[:fft_size]))
        print(time.time() - t1)


class fft_calc(object):
    def __init__(self,N=1024,ROUND=None):
        self.N = N
        self.ROUND  = ROUND
        self.k_arr  = np.arange(self.N)
        self.Wn_arr = np.exp(-2j*np.pi/self.N * np.arange(self.N+1))

        self.log2N = int(np.log2(N))#taking only powers of 2 for now 
        
        self.lookup_8bit_rev = [eval('0b'+bin(i)[2:].zfill(         8)[::-1]) for i in range(1<<8)     ]
        self.lookup_Nbit_rev = [eval('0b'+bin(i)[2:].zfill(self.log2N)[::-1]) for i in range(1<<self.log2N)] #try a more efficient approach

        mask_bf1 = np.zeros((self.log2N, self.N), dtype=bool) #mask for part1 of butterfly
        mask_bf2 = np.zeros((self.log2N, self.N), dtype=bool) #mask for part2 of butterfly

        self.stage_Weights = []

        for i in range(self.log2N):
            mask_bf2[i,((self.k_arr >> i)&1)!=0] = 1 #shouldn't be an issues even if we consider numpy's rotation shifting instead of normal shifting
            mask_bf1[i] = ~mask_bf2[i]
            #print(mask_bf2.sum(),'sum', i)

            rep_cnt = 1<<(self.log2N-i-1)
            weight_lookup_ind = np.tile(np.arange(0,self.N//2,rep_cnt),reps=[rep_cnt])
            #print(weight_lookup_ind, i, mask_bf2)
            stage_Weight = self.Wn_arr[weight_lookup_ind]
            if ROUND is not None:
                if isinstance(ROUND, int):
                    stage_Weight = np.round(stage_Weight,ROUND)
                elif isinstance(ROUND, str) and ROUND.startswith('0b'):
                    stage_Weight = my_bin_round(stage_Weight,eval(ROUND))
                else:
                    assert False

            self.stage_Weights.append(stage_Weight)

        self.mask_bf1 = mask_bf1
        self.mask_bf2 = mask_bf2

    def fft(self,Input):
        stage_Input = Input[self.lookup_Nbit_rev]

        for i in range(self.log2N):

            stage_Output = np.zeros_like(stage_Input, dtype=np.complex128)

            G  = stage_Input[ self.mask_bf1[i]]
            H1 = stage_Input[ self.mask_bf2[i]] * self.stage_Weights[i]

            stage_Output[     self.mask_bf1[i]] = G + H1
            stage_Output[     self.mask_bf2[i]] = G - H1

            stage_Input = stage_Output

        return stage_Output

def my_bin_round(x,ROUND):
    #x : integer, ROUND: the number of binary digits that should be left after rounding
    shift = 1 << ROUND
    return np.round(x * shift)/shift


##manually testing/verifying my_bin_round function
#for i in range(10):
#    x = 0b1101_011_101 / 64; y = my_bin_round(x, i); print(i,bin(int(x*64)),bin(int(y*64)),x,y,abs(y-x))


for i00 in range(17,9,-1):
    #fft_calc_obj = fft_calc(N=(1<<i00), ROUND=None)
    #fft_calc_obj = fft_calc(N=(1<<i00), ROUND=1) 
    fft_calc_obj = fft_calc(N=(1<<i00), ROUND='0b11') 
    t1 = time.time()
    data = np.sin(2*np.pi*0.1*np.arange(fft_calc_obj.N)) * 120
    fft1 = fft_calc_obj.fft(data[:fft_calc_obj.N])
    fft2 =       np.fft.fft(data[:fft_calc_obj.N])
    fft1 /= len(fft1)
    fft2 /= len(fft2)
    fft_diff = fft1 - fft2
    fft_diff_abs = np.abs(fft_diff)
    print(np.allclose(fft1,fft2),fft_diff_abs.mean(),fft_diff_abs.max(), time.time()-t1, i00, flush=True)
#plt.figure()
#plt.subplot(1,2,1);plt.plot(np.abs(fft1))
#plt.subplot(1,2,2);plt.plot(np.abs(fft2)) 
#plt.show()


