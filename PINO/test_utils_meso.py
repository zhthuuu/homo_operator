import torch
import yaml
from matplotlib import pyplot as plt
from train_utils.datasets import HyperElastic_mesoscale
from models import FNN2d
import numpy as np 
from train_utils.losses import calc_Weff_NH_mesoscale, set_kubc, hyper_loss_NH_mesoscale
from torch.utils.data import DataLoader

torch.manual_seed(41)

# test function calc_Weff_NH_mesoscale
# config_file = 'configs/pretrain/mesoscale/random_seed.yaml'
# with open(config_file, 'r') as stream:
#     config = yaml.load(stream, yaml.FullLoader)

# prop = config['prop']
# path = '../data/mesoscale_grid/shear/train_N50_s128.mat'
# nx = 128
# sub = 1
# num = 60
# data = HyperElastic_mesoscale(path, nx, sub, num=50, prop=config['prop'])
# a, u = data[0:10]
# print(a.shape, u.shape)
# weff = calc_Weff_NH_mesoscale(u, a, prop)
# print(weff)

# test function set_kubc
C1 = 1.5
C2 = 1.
C3 = 0.
CauchyGreen = torch.tensor([[C1, C3],[C3, C2]], dtype=torch.float)
F = torch.linalg.cholesky(CauchyGreen)
E = F - torch.eye(2)
u = torch.zeros(3, 128, 128, 2)
u_bc = set_kubc(u, E)
x_left = u_bc[0, :, 0, 0]
x_right = u_bc[0, :, -1, 0]
x_up = u_bc[0, 0, :, 0]
x_down = u_bc[0, -1, :, 0]
print(x_left[:10])
print(x_right[:10])
print(x_up[:10])
print(x_down[:10])
# y_left = u_bc[0, :, 0, 1]
# y_right = u_bc[0, :, -1, 1]
# y_up = u_bc[0, 0, :, 1]
# y_down = u_bc[0, -1, :, 1]
# print(y_left[:10])
# print(y_right[:10])
# print(y_up[:10])
# print(y_down[:10])



# test function hyper_loss_NH_mesoscale
# dataloader = DataLoader(data, batch_size=config['train']['batchsize'])

# model = FNN2d(modes1=config['model']['modes1'],
#                 modes2=config['model']['modes2'],
#                 fc_dim=config['model']['fc_dim'],
#                 layers=config['model']['layers'],
#                 in_dim = 4, out_dim = 2,
#                 activation=config['model']['activation'])

# C1 = 1.5
# C2 = 1.
# C3 = 0.
# CauchyGreen = torch.tensor([[C1, C3],[C3, C2]], dtype=torch.float)
# F = torch.linalg.cholesky(CauchyGreen)
# E = F - torch.eye(2)

# a, u = data[0:10]
# print(a.shape)
# # a[:,:,:,:2] = a[:,:,:,:2]*1e-3
# y_pred = model(a)
# # y_pred = set_kubc(y_pred, E)
# # print(x.shape, y_pred.shape)
# loss = hyper_loss_NH_mesoscale(y_pred, a, E)
# # loss = calc_Weff_NH_mesoscale(u, a)
# print(loss)



