import numpy as np
import torch
import torch.nn.functional as F

def calc_gradN(x, y, device):
    # gradient matrix for FEM 
    G = np.array([[-1/4*(1+y),-1/4*(1-y),1/4*(1-y),1/4*(1+y)],
                  [1/4*(1-x),-1/4*(1-x),-1/4*(1+x),1/4*(1+x)]])
    return torch.tensor(G, device=device)


# ======================= linear elliptic equation loss =====================
def FEM_Elliptic(u, a, xi, L=1):
    # Caulcate Aeff. Integration is calculated with Gaussian quadrature rule
    # define the Gaussian quadrature points
    gpt = 1/np.sqrt(3)
    x = [-gpt, gpt, gpt, -gpt]
    y = [-gpt, -gpt, gpt, gpt]  
    wt = torch.tensor([1., 1., 1., 1.], device=a.device)
    # Jacobian matrix
    s = a.size(1)
    l = L/s
    J = torch.tensor(l/2*np.array([[0,1],[-1,0]]), device=u.device)
    J_inv = torch.inverse(J)
    # u
    u = torch.stack((u[:,1:,:-1], u[:,1:,1:], u[:,:-1,1:], u[:,:-1,:-1]), dim=3)
    u = u.reshape(-1, s*s, 4) # batch x (s*s) x 4
    # calculate Aeff
    batch = a.size(0)
    a_vec = a.reshape(-1, s*s)
    Aeff = torch.zeros(batch, device=a.device)
    for i in range(len(x)):
        B = torch.mm(J_inv, calc_gradN(x[i], y[i], a.device)).float() # 2x4
        gradu = torch.tensordot(u, B, dims=([2],[1])) # batch x (s*s), 2
        tmp_norm = torch.square(torch.norm(gradu+xi, p=2, dim=2)) # batch x (s*s)
        tmp_au = torch.sum(a_vec * tmp_norm, dim=1).squeeze()
        Aeff = Aeff+wt[i]*torch.det(J)*tmp_au
    return Aeff


def elliptic_loss(u, a):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    # expand u with periodic BC
    u = torch.cat([u[:,-1,:].unsqueeze(1), u], dim=1)
    u = torch.cat([u, u[:,:,0].unsqueeze(2)], dim=2)
    mseloss = torch.nn.MSELoss()
    xi = torch.tensor([1., 1.], device=u.device).reshape(1,2)
    xi = xi / torch.norm(xi)
    Du = FEM_Elliptic(u, a, xi)
    f = torch.zeros(Du.shape, device=u.device)
    loss_f = torch.sqrt(mseloss(Du, f))
    return loss_f

# =================== hyper elastic loss function =====================
def hyper_loss(u, a, E, prop):
    E = E.to(u.device)
    prop = prop.to(u.device)
    u = set_kubc(u, E) # set the kubc on u
    Weff = calc_Weff(u, a, prop)
    # mean square error loss function
    mseloss = torch.nn.MSELoss()
    f = torch.zeros(Weff.shape, device=u.device)
    loss_f = torch.sqrt(mseloss(Weff, f))
    return loss_f

def hyper_loss_NH_mesoscale(u, a, E, prop):
    E = E.to(u.device)
    u = set_kubc(u, E) # set the kubc on u
    Weff = calc_Weff_NH_mesoscale(u, a, prop)
    # mean square error loss function
    mseloss = torch.nn.MSELoss()
    f = torch.zeros(Weff.shape, device=u.device)
    loss_f = torch.sqrt(mseloss(Weff, f))
    # loss_f = mseloss(Weff, f)
    return loss_f

def calc_Weff_NH_mesoscale(u, a, prop):
    # area per grid
    Nele = a.size(1)
    Ae = 1/(Nele * Nele)
    F = calc_F(u, 0, 0)
    J = F[:,:,0]*F[:,:,3] - F[:,:,1]*F[:,:,2]
    J[J<0] = 1e-8 # J must be strictly positive
    C00 = F[:,:,0]*F[:,:,0]+F[:,:,2]*F[:,:,2]
    C11 = F[:,:,1]*F[:,:,1]+F[:,:,3]*F[:,:,3]
    # properties
    bulk_vec = a[:,:,:,0].reshape(-1, Nele*Nele)
    shear_vec = a[:,:,:,1].reshape(-1, Nele*Nele)
    bulk_mean = prop['bulk_mean']
    shear_mean = prop['shear_mean']
    bulk_height = prop['bulk_height']
    shear_height = prop['shear_height']
    bulk_vec = 2*bulk_vec*bulk_height-bulk_height+bulk_mean
    shear_vec = 2*shear_vec*shear_height-shear_height+shear_mean
    # weff, Neo-Hookean material
    Weff = shear_vec/2 * (C00 + C11 - 2) + \
           bulk_vec/2*(J-1)*(J-1) - shear_vec*torch.log(J)
    # Weff = C11
    Weff = Ae * torch.sum(Weff, dim=1)
    return Weff

def calc_Weff(u, a, prop):
    # area per grid
    Nele = a.size(1)
    Ae = 1/(Nele * Nele)
    # calcualte W_eff
    batchsize = u.size(0)
    Weff_total = torch.zeros(batchsize, device=u.device)
    F = calc_F(u, 0, 0)
    # J = det(F)
    J = F[:,:,0]*F[:,:,3] - F[:,:,1]*F[:,:,2]
    # J must be strictly positive
    J[J<0] = 1e-8
    # C = F * F.transpose()
    C00 = F[:,:,0]*F[:,:,0]+F[:,:,2]*F[:,:,2]
    C01 = F[:,:,0]*F[:,:,1]+F[:,:,2]*F[:,:,3]
    C10 = F[:,:,0]*F[:,:,1]+F[:,:,2]*F[:,:,3]
    C11 = F[:,:,1]*F[:,:,1]+F[:,:,3]*F[:,:,3]
    # properties
    a_vec = a.reshape(-1, Nele*Nele)
    alpha1 = torch.zeros(a_vec.shape, device=a.device)
    beta1 = torch.zeros(a_vec.shape, device=a.device)
    s1 = torch.zeros(a_vec.shape, device=a.device)
    alpha1[a_vec==0] = prop[0,0]
    alpha1[a_vec==1] = prop[1,0]
    beta1[a_vec==0] = prop[0,1]
    beta1[a_vec==1] = prop[1,1]
    s1[a_vec==0] = prop[0,2]
    s1[a_vec==1] = prop[1,2]
    s2 = 2*(alpha1 + 2*beta1)
    # weff, Mooney-Riveley material
    Weff = alpha1 * (C00 + C11 - 2) + \
            beta1 * (C00+C11+C00*C11-C01*C10-3) + \
            s1*(J-1)*(J-1)/2 - s2*torch.log(J)
    Weff = Ae * torch.sum(Weff, dim=1)
    return Weff


def set_kubc(u, E):
    # set kubc for u matrix, and expand u from s -> s+1
    # kubc: u = E \dot x
    E = E.to(u.device)
    batchsize = u.size(0)
    s = u.size(1) # u: batchsize x s x s x 2
    x = torch.linspace(0, 1, s+1, device=u.device).reshape([-1,1])
    y = torch.linspace(0, 1, s+1, device=u.device).reshape([-1,1])
    x0 = torch.zeros(1, s+1, device=u.device).reshape([-1,1])
    y0 = torch.zeros(1, s, device=u.device).reshape([-1,1])
    # expand u from s*s to (s+1)*(s+1)
    right_u = torch.cat([y0, y0], dim=1)
    right_u = right_u.reshape([s,1,2]).expand(batchsize, -1, -1, -1)
    u = torch.cat([u, right_u], dim=2) # right
    down_u = torch.cat([x0, x0], dim=1)
    down_u = down_u.reshape([1,s+1,2]).expand(batchsize, -1, -1, -1)
    u = torch.cat([u, down_u], dim=1) # down
    # set KUBC
    left_u = torch.cat([E[0,1]*y, E[1,1]*y], dim=1)
    right_u = torch.cat([E[0,0]+E[0,1]*y, E[1,0]+E[1,1]*y], dim=1)
    up_u = torch.cat([E[0,0]*x+E[0,1], E[1,0]*x+E[1,1]], dim=1)
    down_u = torch.cat([E[0,0]*x, E[1,0]*x], dim=1)
    u[:,:,0,:] = left_u # left
    u[:,:,-1,:] = right_u # right
    u[:,0,:,:] = down_u # down
    u[:,-1,:,:] = up_u # up
    return u # batchsize x s+1 x s+1 x2


def calc_F(u, x, y):
    # calclate the gradient field: [u_x,x, u_x,y; u_y,x, u_y,y]
    # return: [u,x, u,y]
    # Jacobian matrix
    s = u.size(1) - 1 # number of elements per edge
    l = 1/s
    J = torch.tensor(l/2*np.array([[0,1],[-1,0]]), device=u.device)
    J_inv = torch.inverse(J)
    ux = torch.stack((u[:,1:,:-1,0], u[:,1:,1:,0], u[:,:-1,1:,0], u[:,:-1,:-1,0]), dim=3)
    uy = torch.stack((u[:,1:,:-1,1], u[:,1:,1:,1], u[:,:-1,1:,1], u[:,:-1,:-1,1]), dim=3)
    # ux = torch.stack((u[:,:-1,:-1,0], u[:,:-1,1:,0], u[:,1:,1:,0], u[:,1:,:-1,0]), dim=3)
    # uy = torch.stack((u[:,:-1,:-1,1], u[:,:-1,1:,1], u[:,1:,1:,1], u[:,1:,:-1,1]), dim=3)
    ux = torch.flip(ux, [3])
    uy = torch.flip(uy, [3])
    ux = ux.reshape(-1, s*s, 4) # batchsize x (s*s) x 4
    uy = uy.reshape(-1, s*s, 4) # batchsize x (s*s) x 4
    batch = u.size(0)
    B = torch.mm(J_inv, calc_gradN(x, y, u.device)).float() # 2x4
    gradux = torch.tensordot(ux, B, dims=([2],[1])) # batchsize x (s*s) x 2
    graduy = torch.tensordot(uy, B, dims=([2],[1])) # batchsize x (s*s) x 2
    I = torch.tensor([1,0,0,1]).reshape([1,4]).float().to(u.device)
    F = torch.cat([gradux, graduy], dim=2) + I
    return F

def calc_C(u):
    F = calc_F(u, 0, 0)
    # C = F * F.transpose()
    # C00 = F[:,:,0]*F[:,:,0]+F[:,:,1]*F[:,:,1]
    # C01 = F[:,:,0]*F[:,:,2]+F[:,:,1]*F[:,:,3]
    # C10 = F[:,:,0]*F[:,:,2]+F[:,:,1]*F[:,:,3]
    # C11 = F[:,:,2]*F[:,:,2]+F[:,:,3]*F[:,:,3]
    C00 = F[:,:,0]*F[:,:,0]+F[:,:,2]*F[:,:,2]
    C01 = F[:,:,0]*F[:,:,1]+F[:,:,2]*F[:,:,3]
    C10 = F[:,:,0]*F[:,:,1]+F[:,:,2]*F[:,:,3]
    C11 = F[:,:,1]*F[:,:,1]+F[:,:,3]*F[:,:,3]
    C = torch.cat([C00, C01, C10, C11], dim=0)
    return C.transpose(0,1)


class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs_vec(self, x, y):
        N = x.shape[0]
        all_norms = torch.norm(x - y, self.p, 0)
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

