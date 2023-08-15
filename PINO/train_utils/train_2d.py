import numpy as np
import torch
from tqdm import tqdm
from .utils import save_checkpoint
from .losses import LpLoss, elliptic_loss, hyper_loss, hyper_loss_NH_mesoscale
try:
    import wandb
except ImportError:
    wandb = None

# ======================== elliptic equation solver =================================
def train_2d_elliptic(model,
                      train_loader,
                      optimizer, scheduler,
                      config,
                      rank=0, log=False,
                      project='PINO-2d-default',
                      group='default',
                      tags=['default'],
                      use_tqdm=True,
                      profile=False):
    '''
    train PINO on inhomogeneous Elliptic equation

    '''
    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity='hzzheng-pino',
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    mesh = train_loader.dataset.mesh
    for e in pbar:
        loss_dict = {'train_loss': 0.0,
                     'data_loss': 0.0,
                     'f_loss': 0.0,
                     'test_error': 0.0}
        for x, y in train_loader:
            x, y = x.to(rank), y.to(rank)

            optimizer.zero_grad()
            pred = model(x).reshape(y.shape)

            # data_loss = myloss(pred, y)
            a = x[..., 0]

            f_loss = elliptic_loss(pred, a)

            loss = f_weight * f_loss
            loss.backward()
            optimizer.step()

            loss_dict['train_loss'] += loss.item() * y.shape[0]
            loss_dict['f_loss'] += f_loss.item() * y.shape[0]
            # loss_dict['data_loss'] += data_loss.item() * y.shape[0]

        scheduler.step()
        train_loss_val = loss_dict['train_loss'] / len(train_loader.dataset)
        f_loss_val = loss_dict['f_loss'] / len(train_loader.dataset)
        # data_loss_val = loss_dict['data_loss'] / len(train_loader.dataset)


        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch: {e}, train loss: {train_loss_val:.5f}, '
                    f'f_loss: {f_loss_val:.5f}, '
                    # f'data loss: {data_loss_val:.5f}, '
                )
            )
        if wandb and log:
            wandb.log(
                {
                    'train loss': train_loss_val,
                    'f loss': f_loss_val,
                    # 'data loss': data_loss_val,
                }
            )
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    if wandb and log:
        run.finish()
    print('Done!')


# ============== hyperelastic solver, Mooney-Rivlin material, microscale ======================
def train_2d_hyper(model, prop, E,
                   train_loader,
                   optimizer, scheduler,
                   config,
                   rank=0, log=False,
                   project='PINO-2d-default',
                   group='default',
                   tags=['default'],
                   use_tqdm=True,
                   profile=False):
    '''
    train PINO for homogenization of hyperelastic material

    '''
    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity='hyper-pino',
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    # mesh = train_loader.dataset.mesh
    loss_info = []
    for e in pbar:
        loss_dict = {'train_loss': 0.0,
                     'data_loss': 0.0,
                     'f_loss': 0.0,
                     'test_error': 0.0}
        for x, y in train_loader:
            x, y = x.to(rank), y.to(rank)

            optimizer.zero_grad()
            pred = model(x)

            # data_loss = myloss(pred, y)
            a = x[..., 0]

            f_loss = hyper_loss(pred, a, E, prop)

            loss = f_weight * f_loss
            loss.backward()
            optimizer.step()

            loss_dict['train_loss'] += loss.item() * x.shape[0]
            loss_dict['f_loss'] += f_loss.item() * x.shape[0]
            # loss_dict['data_loss'] += data_loss.item() * y.shape[0]

        scheduler.step()
        train_loss_val = loss_dict['train_loss'] / len(train_loader.dataset)
        f_loss_val = loss_dict['f_loss'] / len(train_loader.dataset)
        # data_loss_val = loss_dict['data_loss'] / len(train_loader.dataset)
        loss_info.append(train_loss_val)

        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch: {e}, train loss: {train_loss_val:.5f}, '
                    f'f_loss: {f_loss_val:.5f}, '
                    # f'data loss: {data_loss_val:.5f}, '
                )
            )
        if wandb and log:
            wandb.log(
                {
                    'train loss': train_loss_val,
                    'f loss': f_loss_val,
                    # 'data loss': data_loss_val,
                }
            )
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    if wandb and log:
        run.finish()
    # save loss info
    loss_info = np.asarray(loss_info)
    np.save('loss_info/hyper_elastic/training.npy', loss_info)
    print('Done!')


# ==================== hyperelastic solver Neo-Hookean material, mesoscale ====================
def train_2d_hyper_mesoscale(model, E,
                            train_loader,
                            optimizer, scheduler,
                            config,
                            rank=0, log=False,
                            project='default',
                            group='default',
                            tags=['default'],
                            use_tqdm=True,
                            save_name = None):
    
    if wandb and log:
        run = wandb.init(project=project,
                         group=group,
                         config=config,
                         tags=tags, reinit=True)

    f_weight = config['train']['f_loss']
    prop = config['prop']
    model.train()
    pbar = range(config['train']['epochs'])
    if use_tqdm: pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    loss_info = []
    for e in pbar:
        loss_dict = {'train_loss': 0.0,
                     'f_loss': 0.0,
                     'test_error': 0.0}
        for x, y in train_loader:
            x, y = x.to(rank), y.to(rank)
            optimizer.zero_grad()
            y_pred = model(x)
            f_loss = hyper_loss_NH_mesoscale(y_pred, x, E, prop)
            loss = f_weight * f_loss
            loss.backward()
            optimizer.step()
            loss_dict['train_loss'] += loss.item() * x.shape[0]
            loss_dict['f_loss'] += f_loss.item() * x.shape[0]

        scheduler.step()
        train_loss_val = loss_dict['train_loss'] / len(train_loader.dataset)
        f_loss_val = loss_dict['f_loss'] / len(train_loader.dataset)
        f_loss_val = f_loss_val
        loss_info.append(train_loss_val)

        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch: {e}, train loss: {train_loss_val:.5f}, '
                    f'f_loss: {f_loss_val:.5f}, '
                )
            )
        if wandb and log:
            wandb.log(
                {
                    'train loss': train_loss_val,
                    'f loss': f_loss_val,
                }
            )
    if save_name == None: save_name = config['train']['save_name']
    save_checkpoint(config['train']['save_dir'],
                    save_name+'.pt',
                    model)
    if wandb and log:
        run.finish()
    # save loss info
    loss_info = np.asarray(loss_info)
    loss_save_path = 'loss_info/' + config['train']['save_dir'] + save_name
    np.save(loss_save_path, loss_info)
    print('Done!')




