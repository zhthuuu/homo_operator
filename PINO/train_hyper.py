import yaml
from argparse import ArgumentParser
import random

import torch

from models import FNN2d
from train_utils import Adam
from torch.utils.data import DataLoader
from train_utils.datasets import HyperElastic
from train_utils.train_2d import train_2d_hyper


def train(args, config, prop, E):
    seed = random.randint(1, 10000)
    print(f'Random seed :{seed}')
    torch.manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    prop = prop.to(device)
    data_config = config['data']
    dataset = HyperElastic(data_config['datapath'],
                        nx=data_config['nx'], sub=data_config['sub'],
                        offset=data_config['offset'], num=data_config['n_sample'])
    dataloader = DataLoader(dataset, batch_size=config['train']['batchsize'])
    model = FNN2d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  out_dim = 2,
                  activation=config['model']['activation']).to(device)
    # Load from checkpoint
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                         lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])
    train_2d_hyper(model, prop, E,
                      dataloader,
                      optimizer, scheduler,
                      config, rank=0, log=args.log,
                      project=config['others']['project'],
                      group=config['others']['group'])


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--start', type=int, help='Start index of test instance')
    parser.add_argument('--stop', type=int, help='Stop index of instances')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

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
    C2 = 1.
    C3 = 0.
    CauchyGreen = torch.tensor([[C1, C3],[C3, C2]], dtype=torch.float)
    F = torch.linalg.cholesky(CauchyGreen)
    E = F - torch.eye(2)
    # train
    train(args, config, prop, E)



