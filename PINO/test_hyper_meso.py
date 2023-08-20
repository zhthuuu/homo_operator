import torch
import yaml
from matplotlib import pyplot as plt
from train_utils.datasets import HyperElastic_mesoscale
from models import FNN2d
import numpy as np 
from train_utils.losses import calc_Weff_NH_mesoscale, set_kubc, hyper_loss_NH_mesoscale
import time
from train_utils.eval_2d import eval_mesoscale

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
model_path = 'checkpoints/' + config['train']['save_dir'] + config['train']['save_name'] + '.pt'
saved_dict = torch.load(model_path, map_location=torch.device('cuda'))
model.load_state_dict(saved_dict['model'])
t3 = time.time()
print('load dataset: {:.1f}s, load model: {:.1f}s'.format(t2-t1, t3-t2))

# deformation tensor
C1 = 1.3
C2 = 1.2
C3 = 0.2
CauchyGreen = torch.tensor([[C1, C3],[C3, C2]], dtype=torch.float)
F = torch.linalg.cholesky(CauchyGreen) - torch.eye(2)
E = torch.transpose(F, 0, 1)

# # evaluation
# eval_mesoscale(model, dataset, prop, E)

# plot
s = 128
lx = 1/s
a, u_pde = dataset[2]
a, u_pde = a.unsqueeze(0), u_pde.unsqueeze(0)
u_nn = model(a)
u_nn = set_kubc(u_nn, E)
[x, y] = np.meshgrid(np.linspace(0,1,s), np.linspace(0,1,s))
[xu, yu] = np.meshgrid(np.linspace(0,1,s+1), np.linspace(0,1,s+1))
a, u_pde = a.squeeze().numpy(), u_pde.squeeze().numpy()
u_nn = u_nn.squeeze().detach().numpy()
u_diff = u_nn - u_pde
plt.figure(figsize=[14, 8])
plt.subplot(2,3,1)
plt.title('a(x)')
plt.pcolor(x, y, a[...,0],shading='auto')
plt.colorbar()
plt.subplot(2,3,2)
plt.title('ux (pde)')
plt.pcolor(xu, yu, u_pde[:,:,0],shading='auto', cmap='jet')
plt.colorbar()
plt.subplot(2,3,3)
plt.title('ux (NN)')
plt.pcolor(xu, yu, u_nn[:,:,0],shading='auto', cmap='jet')
plt.colorbar()
plt.subplot(2,3,4)
plt.title('a(x)')
plt.pcolor(x, y, a[...,1],shading='auto')
plt.colorbar()
plt.subplot(2,3,5)
plt.title('uy (pde)')
plt.pcolor(xu, yu, u_pde[:,:,1],shading='auto', cmap='jet')
plt.colorbar()
plt.subplot(2,3,6)
plt.title('uy (NN)')
plt.pcolor(xu, yu, u_nn[:,:,1],shading='auto', cmap='jet')
plt.colorbar()
plt.show()






