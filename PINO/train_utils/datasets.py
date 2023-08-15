import scipy.io
import numpy as np

try:
    from pyDOE import lhs
    # Only needed for PINN's dataset
except ImportError:
    lhs = None

import torch
from torch.utils.data import Dataset
from .utils import convert_ic, torch2dgrid


def online_loader(sampler, S, T, time_scale, batchsize=1):
    while True:
        u0 = sampler.sample(batchsize)
        a = convert_ic(u0, batchsize,
                       S, T,
                       time_scale=time_scale)
        yield a


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        self.data = scipy.io.loadmat(self.file_path)
        self.old_mat = True

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# read data from linear elliptic equation with PBC
class Elliptic(Dataset):
    def __init__(self,
                 datapath,
                 nx, sub,
                 offset=0,
                 num=1):
        self.S = int(nx // sub) 
        data = scipy.io.loadmat(datapath)
        a = data['coef']
        u = data['sol']
        self.a = torch.tensor(a[offset: offset + num, ::sub, ::sub], dtype=torch.float)
        self.u = torch.tensor(u[offset: offset + num, ::sub, ::sub], dtype=torch.float)
        self.mesh = torch2dgrid(self.S, self.S)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, item):
        fa = self.a[item]
        return torch.cat([fa.unsqueeze(2), self.mesh], dim=2), self.u[item]

# read data from hyper elastic equation with KUBC
class HyperElastic(Dataset):
    def __init__(self,
                 datapath,
                 nx, sub,
                 offset=0,
                 num=1):
        self.S = int(nx // sub) 
        data = scipy.io.loadmat(datapath)
        a = data['a']
        u = data['u']
        self.a = torch.tensor(a[offset: offset + num, ::sub, ::sub], dtype=torch.float)
        self.u = torch.tensor(u[offset: offset + num, ::sub, ::sub], dtype=torch.float)
        self.mesh = torch2dgrid(self.S, self.S)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, item):
        fa = self.a[item]
        return torch.cat([fa.unsqueeze(2), self.mesh], dim=2), self.u[item]
    
# read data from hyper elastic equation with KUBC
class HyperElastic_mesoscale(Dataset):
    def __init__(self,
                 datapath,
                 nx, sub,
                 offset=0,
                 num=1, prop=None):
        self.S = int(nx // sub) 
        data = scipy.io.loadmat(datapath)
        a = data['a']
        u = data['u']
        bulk_mean = prop['bulk_mean']
        shear_mean = prop['shear_mean']
        bulk_height = prop['bulk_height']
        shear_height = prop['shear_height']
        self.a = torch.tensor(a[offset: offset + num, ::sub, ::sub], dtype=torch.float) # size (N x s x s x 2)
        self.u = torch.tensor(u[offset: offset + num, ::sub, ::sub], dtype=torch.float)
        # normalization of a
        self.a[...,0] = (self.a[...,0]-bulk_mean)/2/bulk_height+0.5
        self.a[...,1] = (self.a[...,1]-shear_mean)/2/shear_height+0.5
        self.mesh = torch2dgrid(self.S, self.S)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, item):
        fa = self.a[item]
        if len(fa.shape) == 3:
            return torch.cat([fa, self.mesh], dim=2), self.u[item]
        else:
            mesh = self.mesh.unsqueeze(0).expand(fa.size(0), -1, -1, -1)
            return torch.cat([fa, mesh], dim=3), self.u[item]





