import os


class Logger:
    def __init__(self, filepath, mode, lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode
        self.lock = lock

    def log(self, str, verbose=True):
        if self.lock:
            self.lock.acquire()
        try:
            with open(self.filepath, self.mode) as f:
                f.write(str + '\n')
        except Exception as e:
            print(e)
        if self.lock:
            self.lock.release()
        if verbose:
            print(str)


def set_log(config, foldername, is_train=True):
    data = config.data.data
    exp_name = foldername

    log_folder_name = os.path.join(*[data, exp_name])
    root = 'logs_train' if is_train else 'logs_sample'
    if not(os.path.isdir(f'./{root}/{log_folder_name}')):
        os.makedirs(os.path.join(f'./{root}/{log_folder_name}'))
    log_dir = os.path.join(f'./{root}/{log_folder_name}/')

    if not(os.path.isdir(f'./checkpoints/{data}')) and is_train:
        os.makedirs(os.path.join(f'./checkpoints/{data}'))
    ckpt_dir = os.path.join(f'./checkpoints/{data}/')

    print('-'*100)
    
    return log_folder_name, log_dir, ckpt_dir


def check_log(log_folder_name, log_name):
    return os.path.isfile(f'./logs_sample/{log_folder_name}/{log_name}.log')


def data_log(logger, config):
    logger.log(f'[{config.data.data}] seed={config.seed}')


def sde_log(logger, config_sde):
    sde_x = config_sde.x
    sde_adj = config_sde.adj
    logger.log(f'(X:{sde_x.type})=({sde_x.beta_min:.2f}, {sde_x.beta_max:.2f}) N={sde_x.num_scales} ' 
               f'(A:{sde_adj.type})=({sde_adj.beta_min:.2f}, {sde_adj.beta_max:.2f}) N={sde_adj.num_scales}')


def model_log(logger, config):
    config_m = config.model
    model_log = f'({config_m.x})+({config_m.adj}={config_m.conv},{config_m.num_heads}): '\
                f'depth={config_m.depth} adim={config_m.adim} nhid={config_m.nhid} layers={config_m.num_layers} '\
                f'linears={config_m.num_linears} c=({config_m.c_init} {config_m.c_hid} {config_m.c_final})'
    logger.log(model_log)


def start_log(logger, config):
    logger.log('-'*100)
    logger.log(f'{config.exp_name}')
    logger.log('-'*100)
    data_log(logger, config)
    logger.log('-'*100)


def train_log(logger, config):
    logger.log(f'lr={config.train.lr} schedule={config.train.lr_schedule} '
               f'epochs={config.train.num_epochs} reduce={config.train.reduce_mean} eps={config.train.eps}')
    sde_log(logger, config.sde)
    model_log(logger, config)
    logger.log('-'*100)


def sample_log(logger, config):
    for model in config.model.keys():
        sample_log = f"[{model}] ({config.model[model].predictor})+({config.model[model].corrector}) "
        if config.model[model].corrector == 'Langevin':
            sample_log += f'snr={config.model[model].snr} seps={config.model[model].scale_eps} '\
                            f'n_steps={config.model[model].n_steps}'
        logger.log(sample_log)
        logger.log('-'*100)
