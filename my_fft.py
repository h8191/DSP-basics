import time
import numpy as np
import matplotlib.pyplot as plt

data = np.random.randint(-1,2,32768)*64

def my_fft(data):
    return data

for fft_func in [np.fft.fft, my_fft]:
    for fft_size in [1024,8192,32768]:
        t1 = time.time()
        f_abs = np.abs(fft_func(data[:fft_size]))
        print(time.time() - t1)


class fft_calc(object):
    def __init__(self,N=1024):
        self.N = N
        self.k_arr  = np.arange(self.N)
        self.Wn_arr = np.exp(-2j*np.pi/self.N * np.arange(self.N+1))

        self.log2N = int(np.log2(N))#taking only powers of 2 for now 
        
        self.lookup_8bit_rev = [eval('0b'+bin(i)[2:].zfill(         8)[::-1]) for i in range(1<<8)     ]
        self.lookup_Nbit_rev = [eval('0b'+bin(i)[2:].zfill(self.log2N)[::-1]) for i in range(1<<self.log2N)] #try a more efficient approach

    def fft(self,Input,ROUND=None):
        stage_Input = Input[self.lookup_Nbit_rev]

        for i in range(self.log2N):
            mask_bf1 = np.zeros(self.N, dtype=bool) #mask for part1 of butterfly
            mask_bf2 = np.zeros(self.N, dtype=bool) #mask for part2 of butterfly

            stage_Output = np.zeros_like(stage_Input, dtype=np.complex128)

            mask_bf2[((self.k_arr >> i)&1)!=0] = 1 #shouldn't be an issues even if we consider numpy's rotation shifting instead of normal shifting
            #print(mask_bf2.sum(),'sum', i)
            mask_bf1 = ~mask_bf2

            rep_cnt = 1<<(self.log2N-i-1)
            weight_lookup_ind = np.tile(np.arange(0,self.N//2,rep_cnt),reps=[rep_cnt])
            #print(weight_lookup_ind, i, mask_bf2)
            stage_Weight = self.Wn_arr[weight_lookup_ind]
            if ROUND is not None:
                stage_Weight = np.round(stage_Weight,ROUND)

            G  = stage_Input[ mask_bf1]
            H1 = stage_Input[ mask_bf2] * stage_Weight

            stage_Output[ mask_bf1] = G + H1
            stage_Output[ mask_bf2] = G - H1

            stage_Input = stage_Output

        return stage_Output

fft_calc_obj = fft_calc(N=(1<<10))
data = np.sin(2*np.pi*0.1*np.arange(fft_calc_obj.N)) * 120
fft1 = fft_calc_obj.fft(data[:fft_calc_obj.N], ROUND=1) 
fft2 =       np.fft.fft(data[:fft_calc_obj.N])
fft1 /= len(fft1)
fft2 /= len(fft2)
fft_diff = fft1 - fft2
fft_diff_abs = np.abs(fft_diff)
print(np.allclose(fft1,fft2),fft_diff_abs.mean(),fft_diff_abs.max(),flush=True)
plt.figure()
plt.subplot(1,2,1);plt.plot(np.abs(fft1))
plt.subplot(1,2,2);plt.plot(np.abs(fft2)) 
plt.show()

#print(fft_calc_obj.lookup_8bit_rev)
#print(fft_calc_obj.log2N)
#t1 = time.time()
#fft_calc_obj.fft(data[:fft_calc_obj.N])
#print(time.time()-t1)
##plt.plot(np.abs(np.fft.fft(data))/len(data));plt.show()
