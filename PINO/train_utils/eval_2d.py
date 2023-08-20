from tqdm import tqdm
import numpy as np

import torch

from .losses import calc_Weff_NH_mesoscale, set_kubc, calc_F

try:
    import wandb
except ImportError:
    wandb = None


def eval_mesoscale(model, dataset, prop, E):
    model.eval()
    with torch.no_grad():
        a, u = dataset[0:-1]
        u_pred = model(a)
        u_pred = set_kubc(u_pred, E)
        REE = calc_REE(u, u_pred, a, prop)
        RHE = calc_RHE(u, u_pred)
        print('relative Weff error (REE) is {:.5f}'.format(REE))
        print('relative H1 error (RHE) is {:.5f}'.format(RHE))

def calc_REE(u, u_pred, a, prop):
    Weff_true = calc_Weff_NH_mesoscale(u, a, prop)
    Weff_pred = calc_Weff_NH_mesoscale(u_pred, a, prop)
    N = Weff_true.shape[0]
    REE = 1/N*torch.sum(((Weff_pred-Weff_true)**2/(Weff_true)**2)**0.5)
    return REE

def calc_RHE(u, u_pred):
    # gradient of u
    F = calc_F(u, 0, 0)
    F_pred = calc_F(u_pred, 0, 0)
    F_diff_norm = torch.norm(F-F_pred, dim=[1,2])**2
    F_norm = torch.norm(F, dim=[1,2])**2
    # u
    u_norm = torch.norm(u, dim=[1,2])
    u_norm = torch.norm(u_norm, dim=[1])**2
    u_diff = u-u_pred
    u_diff_norm = torch.norm(u_diff, dim=[1,2])
    u_diff_norm = torch.norm(u_diff_norm, dim=[1])**2
    N = u.shape[0]
    RHE = 1/N*torch.sum(((u_diff_norm+F_diff_norm)/(u_norm+F_norm))**(1/2))
    return RHE




