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

# deformation tensor
C1 = 1.5
C2 = 1.
C3 = 0.
CauchyGreen = torch.tensor([[C1, C3],[C3, C2]], dtype=torch.float)
F = torch.linalg.cholesky(CauchyGreen)
E = F - torch.eye(2)


for width in [24, 48, 64, 80]:
    for modes in [5, 10, 20, 30]:
        # load trained model
        modes1 = [modes] * 3
        layers = [width] * 4
        model = FNN2d(modes1=modes1,
                modes2=modes1,
                fc_dim=config['model']['fc_dim'],
                layers=layers,
                in_dim = 4, out_dim = 2,
                activation=config['model']['activation'])
        model_path = 'checkpoints/mesoscale/width_modes/' + str(width) + '_' + str(modes) + '.pt'
        saved_dict = torch.load(model_path, map_location=torch.device('cuda'))
        model.load_state_dict(saved_dict['model'])

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
        print('width={}, modes={}, REE={:.5f}'.format(width, modes, REE))

# results:
# width=24, modes=5, REE=0.02434
# width=24, modes=10, REE=0.01815
# width=24, modes=20, REE=0.02634
# width=24, modes=30, REE=0.04357
# width=48, modes=5, REE=0.00972
# width=48, modes=10, REE=0.01133
# width=48, modes=20, REE=0.01817
# width=48, modes=30, REE=0.02458
# width=64, modes=5, REE=0.01245
# width=64, modes=10, REE=0.01695
# width=64, modes=20, REE=0.01231
# width=64, modes=30, REE=0.02048
# width=80, modes=5, REE=0.00745
# width=80, modes=10, REE=0.01393
# width=80, modes=20, REE=0.01391
# width=80, modes=30, REE=0.01428







