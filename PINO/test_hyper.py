import torch
import yaml
from matplotlib import pyplot as plt
from train_utils.datasets import HyperElastic
from models import FNN2d
import numpy as np 
from train_utils.losses import calc_Weff, set_kubc
from scipy import stats


# config_file = 'configs/pretrain/Hyper_pretrain.yaml'
config_file = 'configs/test/hyper_test.yaml'
with open(config_file, 'r') as stream:
    config = yaml.load(stream, yaml.FullLoader) 
# load trained model
device = torch.device('cpu')
model = FNN2d(modes1=config['model']['modes1'],
              modes2=config['model']['modes2'],
              fc_dim=config['model']['fc_dim'],
              layers=config['model']['layers'],
              out_dim = 2,
              activation=config['model']['activation']).to(device)
saved_dict = torch.load('checkpoints/hyper_elastic/pretrain-pino-ep200-ux.pt', map_location=torch.device('cpu'))
model.load_state_dict(saved_dict['model'])
# load training dataset
data_config = config['data']
dataset = HyperElastic(data_config['datapath'],
                    nx=data_config['nx'], sub=data_config['sub'],
                    offset=data_config['offset'], num=data_config['n_sample'])


# Matrix properties
alpham = 0.5*328.1250    # Alpha in MPa
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

# deformation matrix, u_bc = E \dot x
C1 = 1.5
C2 = 1.0
C3 = 0.0
CauchyGreen = torch.tensor([[C1, C3],[C3, C2]], dtype=torch.float)
F = torch.linalg.cholesky(CauchyGreen)
E = F - torch.eye(2)

# data
sample = 100
a, u_pde = dataset[sample]
a, u_pde = a.unsqueeze(0), u_pde.unsqueeze(0)
u_nn = model(a)
u_nn = set_kubc(u_nn, E) # add Dirichlet boundary condition
a = a[:,:,:,0]

# Weff
Weff_pde = calc_Weff(u_pde, a, prop)[0]
Weff_nn = calc_Weff(u_nn, a, prop)[0]
err = (Weff_nn-Weff_pde)/Weff_pde * 100
print('Weff_pde = {:.4f}, Weff_nn={:.4f}, err={:.4f}%'.format(Weff_pde, Weff_nn, err))

# plot
# s = 64
# lx = 1/s
# [x, y] = np.meshgrid(np.linspace(0,1,s), np.linspace(0,1,s))
# [xu, yu] = np.meshgrid(np.linspace(0,1,s+1), np.linspace(0,1,s+1))
# a, u_pde = a.squeeze().numpy(), u_pde.squeeze().numpy()
# u_nn = u_nn.squeeze().detach().numpy()
# plt.figure(figsize=[14, 8])
# plt.subplot(2,3,1)
# plt.title('a(x)')
# plt.pcolor(x, y, a)
# plt.colorbar()
# plt.subplot(2,3,2)
# plt.title('ux (pde)')
# plt.pcolor(xu, yu, u_pde[:,:,0], cmap='jet')
# plt.colorbar()
# plt.subplot(2,3,3)
# plt.title('ux (NN)')
# plt.pcolor(xu, yu, u_nn[:,:,0], cmap='jet')
# plt.colorbar()
# plt.subplot(2,3,5)
# plt.title('uy (pde)')
# plt.pcolor(xu, yu, u_pde[:,:,1], cmap='jet')
# plt.colorbar()
# plt.subplot(2,3,6)
# plt.title('uy (NN)')
# plt.pcolor(xu, yu, u_nn[:,:,1], cmap='jet')
# plt.colorbar()
# plt.show()



# accuracy of Aeff
# N_sample = data_config['n_sample']
# err = []
# for i in range(N_sample):
#     a, u_pde = dataset[i]
#     a, u_pde = a.unsqueeze(0), u_pde.unsqueeze(0)
#     u_nn = model(a)
#     u_nn = set_kubc(u_nn, E) # add Dirichlet boundary condition
#     a = a[:,:,:,0]
#     Weff_pde = calc_Weff(u_pde, a, prop)[0]
#     Weff_nn = calc_Weff(u_nn, a, prop)[0]
#     err_i  = (Weff_nn-Weff_pde)/Weff_pde * 100
#     err.append(err_i.item())
#     if (i+1) % 100 == 0:
#         print('{}/{} samples calculated.'.format(i+1, N_sample))
# np.save('data/hyper_elastic/test_err.npy', np.array(err))


