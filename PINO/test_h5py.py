import numpy as np 
import h5py 
import scipy.io 
import time 

file_name = '../data/mesoscale_grid/s128/test_N1000_s128.mat'
file_name2 = '../data/mesoscale_grid/s128/test_N1000_s128_v0.mat'
t1 = time.time()
f1 = h5py.File(file_name, 'r')
a = np.array(f1['a'])
a = np.transpose(a, [3,2,1,0])
print(a.shape)
t2 = time.time()

f2 = scipy.io.loadmat(file_name)
b = f2['a']
t3 = time.time()

print(t2-t1, t3-t2)

print(np.linalg.norm(a[0,:,:,0]-b[0,:,:,0]))

