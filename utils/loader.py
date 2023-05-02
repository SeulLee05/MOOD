import torch
import random
import numpy as np

from models.ScoreNetwork_A import ScoreNetworkA
from models.ScoreNetwork_X import ScoreNetworkX, ScoreNetworkX_GMH
from losses import get_sde_loss_fn

def load_seed(seed):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


def load_device(gpu):
    if isinstance(gpu, int):
        gpu = str(gpu)
    device = [int(g) for g in gpu.split(',')]
    return device   # list of integers


def load_model(params):
    params_ = params.copy()
    model_type = params_.pop('model_type', None)
    if model_type == 'ScoreNetworkX':
        model = ScoreNetworkX(**params_)
    elif model_type == 'ScoreNetworkX_GMH':
        model = ScoreNetworkX_GMH(**params_)
    elif model_type == 'ScoreNetworkA':
        model = ScoreNetworkA(**params_)
    else:
        raise ValueError(f"Model Name <{model_type}> is Unknown")
    return model


def load_model_optimizer(params, config_train, device):
    if len(device) == 1:
        model = load_model(params)
    else:
        model = torch.nn.DataParallel(load_model(params), device_ids=device)
    model = model.to(f'cuda:{device[0]}')
    optimizer = torch.optim.Adam(model.parameters(), lr=config_train.lr, 
                                 weight_decay=config_train.weight_decay)
    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    
    return model, optimizer, scheduler


def load_data(config, get_graph_list=False):
    from utils.data_loader_mol import dataloader
    return dataloader(config, get_graph_list)


def load_prop_data(config):
    assert config.data.data == 'ZINC250k'
    from utils.data_loader_mol import dataloader
    return dataloader(config, get_graph_list=False, prop=config.train.prop)


def load_batch(batch, device):
    x_b = batch[0].to(f'cuda:{device[0]}')
    adj_b = batch[1].to(f'cuda:{device[0]}')

    return x_b, adj_b


def load_sde(config_sde):
    from sde import VPSDE, VESDE, subVPSDE
    sde_type = config_sde.type
    beta_min = config_sde.beta_min
    beta_max = config_sde.beta_max
    num_scales = config_sde.num_scales

    if sde_type == 'VP':
        sde = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    elif sde_type == 'subVP':
        sde = subVPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    elif sde_type == 'VE':
        sde = VESDE(sigma_min=beta_min, sigma_max=beta_max, N=num_scales)
    else:
        raise NotImplementedError(f"SDE class {sde_type} not yet supported.")

    return sde


def load_loss_fn(config):
    reduce_mean = config.train.reduce_mean
    sde_x = load_sde(config.sde.x)
    sde_adj = load_sde(config.sde.adj)
    
    loss_fn = get_sde_loss_fn(sde_x, sde_adj, train=True, reduce_mean=reduce_mean, continuous=True, 
                              likelihood_weighting=False, eps=config.train.eps)
    return loss_fn


def load_sampling_fn(config_train, config_module, config_classifier,
                     config_sample, device):
    snr = config_module.snr
    scale_eps = config_module.scale_eps

    sde_x = load_sde(config_train.sde.x)
    sde_adj = load_sde(config_train.sde.adj)
    
    from solver import get_pc_sampler
    get_sampler = get_pc_sampler
    
    shape_x = (config_sample.num_samples, config_train.data.max_node_num, config_train.data.max_feat_num)
    shape_adj = (config_sample.num_samples, config_train.data.max_node_num, config_train.data.max_node_num)

    sampling_fn = get_sampler(sde_x=sde_x, sde_adj=sde_adj, shape_x=shape_x, shape_adj=shape_adj,
                              predictor=config_module.predictor, corrector=config_module.corrector,
                              weight_x=config_classifier.weight_x, weight_adj=config_classifier.weight_adj,
                              snr=snr, scale_eps=scale_eps, n_steps=config_module.n_steps,
                              probability_flow=config_sample.probability_flow,
                              continuous=True, denoise=config_sample.noise_removal,
                              eps=config_sample.eps, device=f'cuda:{device[0]}', ood=config_sample.ood)
    return sampling_fn


def load_model_params(config):
    config_m = config.model
    max_feat_num = config.data.max_feat_num

    if 'GMH' in config_m.x:
        params_x = {'model_type': config_m.x, 'max_feat_num': max_feat_num, 'depth': config_m.depth, 
                    'nhid': config_m.nhid, 'num_linears': config_m.num_linears,
                    'c_init': config_m.c_init, 'c_hid': config_m.c_hid, 'c_final': config_m.c_final, 
                    'adim': config_m.adim, 'num_heads': config_m.num_heads, 'conv':config_m.conv}
    else:
        params_x = {'model_type':config_m.x, 'max_feat_num':max_feat_num, 'depth':config_m.depth, 'nhid':config_m.nhid}
    params_adj = {'model_type':config_m.adj, 'max_feat_num':max_feat_num, 'max_node_num':config.data.max_node_num, 
                    'nhid':config_m.nhid, 'num_layers':config_m.num_layers, 'num_linears':config_m.num_linears, 
                    'c_init':config_m.c_init, 'c_hid':config_m.c_hid, 'c_final':config_m.c_final, 
                    'adim':config_m.adim, 'num_heads':config_m.num_heads, 'conv':config_m.conv}
    return params_x, params_adj


def load_ckpt(config, device):
    ckpt_dict = {}
    path = f'./checkpoints/{config.data.data}/{config.model.diff.ckpt}.pth'
    ckpt = torch.load(path, map_location=f'cuda:{device[0]}')
    print(f'{path} loaded')
    model_config = ckpt['model_config']
    ckpt_dict['diff'] = {'config': model_config, 'params_x': ckpt['params_x'], 'x_state_dict': ckpt['x_state_dict'],
                      'params_adj': ckpt['params_adj'], 'adj_state_dict': ckpt['adj_state_dict']}
    return ckpt_dict


def load_model_from_ckpt(params, state_dict, device):
    model = load_model(params)
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k[7:]: v for k, v in state_dict.items()}  # strip 'module.' at front; for DataParallel models
    model.load_state_dict(state_dict)
    if len(device) > 1:
        model = torch.nn.DataParallel(model, device_ids=device)
    model = model.to(f'cuda:{device[0]}')
    
    return model
