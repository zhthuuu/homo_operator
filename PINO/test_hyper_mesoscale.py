import torch
import yaml
from matplotlib import pyplot as plt
from train_utils.datasets import HyperElastic_mesoscale
from models import FNN2d
import numpy as np 
from train_utils.losses import calc_Weff_NH_mesoscale, set_kubc
from torch.utils.data import DataLoader

path = '/Users/hz/Desktop/homo_operator/data/mesoscale_grid/train_N60_s128.mat'
nx = 128
sub = 1
num = 60
data = HyperElastic_mesoscale(path, nx, sub, num=60)
a, u = data.a, data.u
# print(a.shape, u.shape)
# weff = calc_Weff_NH_mesoscale(u, a)
# print(weff)

C1 = 1.5
C2 = 1.
C3 = 0.
CauchyGreen = torch.tensor([[C1, C3],[C3, C2]], dtype=torch.float)
F = torch.linalg.cholesky(CauchyGreen)
E = F - torch.eye(2)

config_file = 'configs/pretrain/hyperelastic_mesoscale_pretrain.yaml'
with open(config_file, 'r') as stream:
    config = yaml.load(stream, yaml.FullLoader)

dataloader = DataLoader(data, batch_size=config['train']['batchsize'])

model = FNN2d(modes1=config['model']['modes1'],
                modes2=config['model']['modes2'],
                fc_dim=config['model']['fc_dim'],
                layers=config['model']['layers'],
                in_dim = 4, out_dim = 2,
                activation=config['model']['activation'])
x, y = data[0]
x = x.unsqueeze(0)
y_pred = model(x)
y_pred = set_kubc(y_pred, E)
print(x.shape, y_pred.shape)
weff = calc_Weff_NH_mesoscale(y_pred, x)
print(weff)



