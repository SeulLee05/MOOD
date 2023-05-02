import os
import time
from tqdm import tqdm
import numpy as np
import torch

from utils.loader import load_seed, load_device, load_sde, load_prop_data
from utils.logger import Logger, set_log
from utils.regressor_utils import load_regressor_params, load_regressor_batch, load_regressor_optimizer, \
                                   load_regressor_loss_fn, start_log, train_log


class Trainer(object):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config, 'prop')

        self.device = load_device(self.config.gpu)

        self.train_loader, self.test_loader = load_prop_data(self.config)
        self.params = load_regressor_params(self.config)

        load_seed(self.config.seed)
    
    def train(self):
        self.config.exp_name = time.strftime('%b%d-%H:%M:%S', time.gmtime())
        self.ckpt = f'prop_{self.config.exp_name}'
        print('\033[91m' + f'{self.ckpt}' + '\033[0m')

        self.model, self.optimizer, self.scheduler = load_regressor_optimizer(self.params, self.config.train, self.device)
        self.eps = self.config.train.eps
        self.sde_x = load_sde(self.config.sde.x)
        self.sde_adj = load_sde(self.config.sde.adj)

        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}_{self.config.train.prop}.log')), mode='a')
        start_log(logger, self.config)
        train_log(logger, self.config)

        logger.log(str(self.model))
        logger.log('-'*100)

        self.loss_fn = load_regressor_loss_fn(self.config)

        for epoch in range(self.config.train.num_epochs):
            self.train_loss = []
            self.test_loss = []
            self.train_corr = []
            self.test_corr = []
            t_start = time.time()

            self.model.train()
            for _, train_b in enumerate(tqdm(self.train_loader, desc=f'[Epoch {epoch+1}]')):
                x, adj, labels = load_regressor_batch(train_b, self.device)

                self.model.train()
                self.optimizer.zero_grad()
                loss, corr = self.loss_fn(self.model, x, adj, labels)
                loss.backward()

                self.train_loss.append(loss.item())
                self.train_corr.append(corr)

                self.optimizer.step()

            if self.config.train.lr_schedule:
                self.scheduler.step()

            self.model.eval()
            for _, test_b in enumerate(self.test_loader):
                x, adj, labels = load_regressor_batch(test_b, self.device)

                with torch.no_grad():
                    loss, corr = self.loss_fn(self.model, x, adj, labels)
                    self.test_loss.append(loss.item())
                    self.test_corr.append(corr)

            logger.log(f'Epoch: {epoch+1:03d} | {time.time()-t_start:.2f}s | '
                       f'TRAIN loss: {np.mean(self.train_loss):.4e} | '
                       f'TRAIN corr: {np.mean(self.train_corr):.4f} | '
                       f'TEST loss: {np.mean(self.test_loss):.4e} | '
                       f'TEST corr: {np.mean(self.test_corr):.4f}', verbose=False)

        torch.save({
            'model_config': self.config,
            'params' : self.params,
            'state_dict': self.model.state_dict()},
            f'./checkpoints/{self.config.data.data}/{self.ckpt}.pth')
