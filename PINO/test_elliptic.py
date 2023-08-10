import torch
import yaml
from matplotlib import pyplot as plt
from train_utils.datasets import Elliptic
from models import FNN2d
import numpy as np 
from train_utils.losses import elliptic_loss
from scipy import stats


# config_file = 'configs/pretrain/Elliptic_pretrain.yaml'
config_file = 'configs/test/Elliptic.yaml'
with open(config_file, 'r') as stream:
    config = yaml.load(stream, yaml.FullLoader) 
# load trained model
device = torch.device('cpu')
model = FNN2d(modes1=config['model']['modes1'],
              modes2=config['model']['modes2'],
              fc_dim=config['model']['fc_dim'],
              layers=config['model']['layers'],
              activation=config['model']['activation']).to(device)
saved_dict = torch.load('checkpoints/elliptic/pretrain-pino-nodbc-ep200.pt', map_location=torch.device('cpu'))
model.load_state_dict(saved_dict['model'])
# load training dataset
data_config = config['data']
dataset = Elliptic(data_config['datapath'],
                    nx=data_config['nx'], sub=data_config['sub'],
                    offset=data_config['offset'], num=data_config['n_sample'])


# accuracy of Aeff
N_sample = data_config['n_sample']
err = []
for i in range(N_sample):
    a,u_true = dataset[i]
    a = a.to(device)
    u_true = u_true.to(device)
    u = model(a.unsqueeze(0)).squeeze()
    a = a[:,:,0] # size[1,s,s]
    a_eff_true = elliptic_loss(u_true.unsqueeze(0), a)
    a_eff_pred = elliptic_loss(u.unsqueeze(0), a)
    err_i = (a_eff_pred-a_eff_true) / a_eff_true * 100
    err.append(err_i.item())
    if (i+1)%100 == 0:
        print('{}/{} cases calculated'.format(i+1, N_sample))
# print('Aeff true={:.4f}, predicted={:.4f}, err={:.2}%'.format(a_eff_true.item(), a_eff_pred.item(), err))

# statistics
max_err = np.max(err)
min_err = np.min(err)
mean_err = np.mean(err)
print('max_err={:.4f}%, min_err={:.4f}%, mean_err={:.4f}%'.format(max_err, min_err, mean_err))
# histogram
plt.hist(err)
plt.xlabel('error (%)')
plt.grid(True)
plt.show()
# kde = stats.gaussian_kde(err)
# x = np.linspace(min_err*0.3, max_err*1.05, 100)
# p = kde(x)
# plt.figure(figsize=[7,5])
# plt.plot(x, p)
# plt.xlabel('err (%)')
# plt.ylabel('pdf')
# plt.grid(True)
# plt.show()

# contour
# s = 64
# lx = 1/s
# [x, y] = np.meshgrid(range(s), range(s))
# plt.figure(figsize=[14, 3.5])
# plt.subplot(1,3,1)
# plt.title('a(x)')
# a = a.squeeze().detach().numpy()
# plt.pcolor(x, y, a)
# plt.colorbar()
# plt.subplot(1,3,2)
# plt.title('prediction')
# plt.pcolor(x, y, u)
# plt.colorbar()
# plt.subplot(1,3,3)
# plt.title('ground truth')
# plt.pcolor(x, y, u_true)
# plt.colorbar()
# plt.show()



