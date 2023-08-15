import yaml
from argparse import ArgumentParser
import random

import torch

from models import FNN2d
from train_utils import Adam
from torch.utils.data import DataLoader
from train_utils.datasets import HyperElastic_mesoscale
from train_utils.train_2d import train_2d_hyper_mesoscale


def train(args, config, E):
    # seed = random.randint(1, 10000)
    seed = 1008
    print(f'Random seed: {seed}')
    torch.manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    data_config = config['data']
    dataset = HyperElastic_mesoscale(data_config['datapath'],
                        nx=data_config['nx'], sub=data_config['sub'],
                        offset=data_config['offset'], num=data_config['n_sample'], prop=config['prop'])
    dataloader = DataLoader(dataset, batch_size=config['train']['batchsize'])
    width = args.width
    layers = [width]*4
    modes = args.modes
    modes1 = [modes] * 3
    modes2 = [modes] * 3
    model = FNN2d(modes1=modes1,
                  modes2=modes2,
                  fc_dim=config['model']['fc_dim'],
                  layers=layers,
                  in_dim = 4, out_dim = 2,
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
    save_name = str(width) + '_' + str(modes)
    train_2d_hyper_mesoscale(model, E, dataloader,
                            optimizer, scheduler,
                            config, rank=device, log=args.log,
                            project=config['others']['project'],
                            group=config['others']['group'],
                            save_name = save_name,
                            use_tqdm=False)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--width', type=int, help='width of FNO')
    parser.add_argument('--modes', type=int, help='Fourier modes of each Fourier layer')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    # deformation matrix, u_bc = E \dot x
    C1 = 1.5
    C2 = 1.
    C3 = 0.
    CauchyGreen = torch.tensor([[C1, C3],[C3, C2]], dtype=torch.float)
    F = torch.linalg.cholesky(CauchyGreen)
    E = F - torch.eye(2)
    # train
    train(args, config, E)

    # python train_hyper_mesoscale.py --config_path configs/pretrain/hyperelastic_mesoscale_pretrain.yaml



