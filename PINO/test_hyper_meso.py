import torch
import yaml
from matplotlib import pyplot as plt
from train_utils.datasets import HyperElastic_mesoscale
from models import FNN2d
import numpy as np 
from train_utils.losses import calc_Weff_NH_mesoscale, set_kubc, hyper_loss_NH_mesoscale
import time

torch.manual_seed(41)

config_file = 'configs/test/hyperelastic_mesoscale.yaml'
with open(config_file, 'r') as stream:
    config = yaml.load(stream, yaml.FullLoader)

# load dataset
t1 = time.time()
data_config = config['data']
prop = config['prop']
dataset = HyperElastic_mesoscale(data_config['datapath'],
                    nx=data_config['nx'], sub=data_config['sub'],
                    offset=data_config['offset'], num=data_config['n_sample'], prop=prop)
t2 = time.time()

# load trained model
model = FNN2d(modes1=config['model']['modes1'],
                modes2=config['model']['modes2'],
                fc_dim=config['model']['fc_dim'],
                layers=config['model']['layers'],
                in_dim = 4, out_dim = 2,
                activation=config['model']['activation'])
model_path = 'checkpoints/' + config['train']['save_dir'] + config['train']['save_name']
saved_dict = torch.load(model_path, map_location=torch.device('cuda'))
model.load_state_dict(saved_dict['model'])
t3 = time.time()
print('load dataset: {:.1f}s, load model: {:.1f}s'.format(t2-t1, t3-t2))

# deformation tensor
C1 = 1.5
C2 = 1.
C3 = 0.
CauchyGreen = torch.tensor([[C1, C3],[C3, C2]], dtype=torch.float)
F = torch.linalg.cholesky(CauchyGreen)
E = F - torch.eye(2)

# calculate Weff
Weff_true, Weff_pred = [], []
a, u = dataset[0:100]
u_pred = model(a)
u_pred = set_kubc(u_pred, E)
Weff_true = calc_Weff_NH_mesoscale(u, a, prop)
Weff_pred = calc_Weff_NH_mesoscale(u_pred, a, prop)
Weff_true = Weff_true.detach().numpy()
Weff_pred = Weff_pred.detach().numpy()

REE = np.linalg.norm(Weff_pred-Weff_true)/np.linalg.norm(Weff_true)
print('relative Weff error is {:.5f}'.format(REE))







