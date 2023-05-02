import torch
import pandas as pd

from utils.graph_utils import node_flags, mask_x, mask_adjs, gen_noise
from models.regressor import Regressor, get_regressor_fn
from utils.loader import load_sde


def load_regressor_params(config):
    config_m = config.model
    params = {'max_node_num': config.data.max_node_num,
              'max_feat_num': config.data.max_feat_num, 'depth':config_m.depth, 
              'nhid': config_m.nhid, 'dropout': config_m.dropout}

    return params

def load_regressor(params):
    params_ = params.copy()
    model = Regressor(**params_)

    return model


def load_regressor_optimizer(params, config_train, device):
    model = load_regressor(params).to(f'cuda:{device[0]}')
    optimizer = torch.optim.Adam(model.parameters(), lr=config_train.lr,
                                 weight_decay=config_train.weight_decay)
    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    
    return model, optimizer, scheduler


def load_regressor_batch(batch, device):
    x_b = batch[0].to(f'cuda:{device[0]}')
    adj_b = batch[1].to(f'cuda:{device[0]}')
    label_b = batch[2].unsqueeze(-1).to(f'cuda:{device[0]}')

    return x_b, adj_b, label_b


def load_regressor_loss_fn(config):
    sde_x = load_sde(config.sde.x)
    sde_adj = load_sde(config.sde.adj)
    eps = config.train.eps

    def loss_fn(model, x, adj, labels):
        regressor_fn = get_regressor_fn(sde_adj, model)
        flags = node_flags(adj)
        t = torch.rand(adj.shape[0], device=adj.device) * (sde_adj.T - eps) + eps

        z_x = gen_noise(x, flags, sym=False)
        mean_x, std_x = sde_x.marginal_prob(x, t)
        perturbed_x = mean_x + std_x[:, None, None] * z_x
        perturbed_x = mask_x(perturbed_x, flags)

        z_adj = gen_noise(adj, flags, sym=True)
        mean_adj, std_adj = sde_adj.marginal_prob(adj, t)
        perturbed_adj = mean_adj + std_adj[:, None, None] * z_adj
        perturbed_adj = mask_adjs(perturbed_adj, flags)

        pred = regressor_fn(perturbed_x, perturbed_adj, flags, t)
        loss = (pred - labels).pow(2).mean()

        with torch.no_grad():
            df = pd.DataFrame()
            df['pred'] = pred.cpu().detach().numpy().squeeze()
            df['labels'] = labels.cpu().detach().numpy().squeeze()
            corr = df.corr()['pred']['labels']

        return loss, corr

    return loss_fn

def load_regressor_from_ckpt(params, state_dict, device):
    model = load_regressor(params)
    model.load_state_dict(state_dict)
    model = model.to(f'cuda:{device[0]}')

    return model


def load_regressor_ckpt(config, device):
    ckpt_dict = {}
    path = f'./checkpoints/{config.data.data}/{config.model.prop.ckpt}.pth'
    ckpt = torch.load(path, map_location=f'cuda:{device[0]}')
    print(f'{path} loaded')
    ckpt_dict['prop'] = {'config': ckpt['model_config'], 'params': ckpt['params'], 'state_dict': ckpt['state_dict']}
    ckpt_dict['prop']['config']['data']['data'] = config.data.data

    return ckpt_dict


def data_log(logger, config):
    logger.log(f'[{config.data.data}] seed={config.seed} batch_size={config.data.batch_size}')


def sde_log(logger, config_sde):
    sde_x = config_sde.x
    sde_adj = config_sde.adj
    logger.log(f'(X:{sde_x.type})=({sde_x.beta_min:.2f}, {sde_x.beta_max:.2f}) N={sde_x.num_scales} ' 
               f'(A:{sde_adj.type})=({sde_adj.beta_min:.2f}, {sde_adj.beta_max:.2f}) N={sde_adj.num_scales}')

def model_log(logger, config):
    config_m = config.model
    model_log = f'({config_m.model}): ' \
                f'depth={config_m.depth} nhid={config_m.nhid} ' \
                f'dropout={config_m.dropout}'
    logger.log(model_log)


def start_log(logger, config, is_train=True):
    if is_train:
        logger.log('-'*100)
        logger.log(f'{config.exp_name}')
    logger.log('-'*100)
    data_log(logger, config)
    logger.log('-'*100)


def train_log(logger, config):
    sde_log(logger, config.sde)
    model_log(logger, config)
    logger.log('-'*100)


def sample_log(logger, configc):
    logger.log(f'[X] weight={configc.weight_x} [A] weight={configc.weight_adj}')
    logger.log('-'*100)
