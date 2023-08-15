import torch
import numpy as np
from models import FNN2d
import yaml
from train_utils.losses import calc_Weff, calc_F, calc_C
import scipy.io 
from matplotlib import pyplot as plt

alpham = 0.5*328.1250    # Alpha in MPa
# alpham = 1
betam = 10               # Beta in MPa
km = 1750                # Bulk modulus
um = 2*alpham            # Shear modulus
lambdam = km - (2/3)*um  # Lambda
sm = lambdam - 4*betam   # s1 = Lambda - 4*Beta
# Inclusion properties
alphaf = 10*alpham
betaf = 2*betam
kf = 10*km
uf = 2*alphaf 
lambdaf = kf - (2/3)*uf
sf = lambdaf - 4*betaf
# property matrix
prop = torch.tensor([[alpham, betam, sm],[alphaf, betaf, sf]], dtype=torch.float)

# test Weff
sample = 99
s = 64
data = scipy.io.loadmat('data/hyper_elastic/bitension_s65_sample100.mat')
# a = torch.from_numpy(data['a'][sample,:,:]).unsqueeze(0).float()
# u = torch.from_numpy(data['u'][sample,:,:,:]).unsqueeze(0).float()
a = torch.from_numpy(data['a']).reshape(1,s,s).float()
u = torch.from_numpy(data['u']).reshape(1,s+1,s+1,2).float()
Weff = calc_Weff(u, a, prop)
print([Weff])

# test C: validated
# C = calc_C(u)
# s = a.size(1)
# [x, y] = np.meshgrid(range(s), range(s))
# C = C.reshape([s,s,4]).numpy()
# plt.subplot(2,2,1)
# plt.pcolor(x, y, C[:,:,0], cmap='jet')
# plt.title('C11')
# plt.colorbar()
# plt.subplot(2,2,2)
# plt.pcolor(x, y, C[:,:,1], cmap='jet')
# plt.title('C12')
# plt.colorbar()
# plt.subplot(2,2,3)
# plt.pcolor(x, y, C[:,:,2], cmap='jet')
# plt.title('C21')
# plt.colorbar()
# plt.subplot(2,2,4)
# plt.pcolor(x, y, C[:,:,3], cmap='jet')
# plt.title('C22')
# plt.colorbar()
# plt.show()

# test gradu: validated
# gradu = calc_F(u, 0, 0).squeeze()
# s = a.size(1)
# [x, y] = np.meshgrid(range(s), range(s))
# gradu = gradu.reshape([s,s,4]).numpy()
# plt.subplot(2,2,1)
# plt.pcolor(x, y, gradu[:,:,0], cmap='jet')
# plt.colorbar()
# plt.subplot(2,2,2)
# plt.pcolor(x, y, gradu[:,:,1], cmap='jet')
# plt.colorbar()
# plt.subplot(2,2,3)
# plt.pcolor(x, y, gradu[:,:,2], cmap='jet')
# plt.colorbar()
# plt.subplot(2,2,4)
# plt.pcolor(x, y, gradu[:,:,3], cmap='jet')
# plt.colorbar()
# plt.show()


