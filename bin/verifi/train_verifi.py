import torch
import torch.nn as nn 
from src.solver import BaseSolver
from bin.verifi.model import VERIFI

class SolverVERIFI(BaseSolver):
    "Solver for training"
    def __init__(self, gpu_rank, paras, config, rank, mode):
        super().__init__(config, paras, mode)
        # Logger settings
        self.best_eer = {'eer': 20.0}
        self.rank = rank
        self.gpu_rank = gpu_rank
    
    def fetch_data(self, data):
        ''' 
        Move data to device and compute seq length
        '''
        pass
    
    def load_data(self):
        ''' 
        Load data for training/validation, store label, input/output shape
        '''
        pass
    
    def set_model(self):
        ''' 
        Setup VERIFI model and optimizer 
        '''
        init_adamw = self.config['hparas']['optimizer'] == 'AdamW'

        # Model
        if self.paras.gpu and self.mode == 'train':
            torch.cuda.set_device(self.gpu_rank)
            self.model = VERIFI(input_size=self.feat_dim,audio_encoder=self.config['model']['encoder']).cuda(self.gpu_rank)
        
        else:
            self.model = VERIFI(input_size=self.feat_dim,audio_encoder=self.config['model']['encoder']).to(self.device)

        self.verbose(self.model.create_msg())
        model_paras = [{'params': self.model.parameters()}]

        # Losses
        self.verifi_loss = self.model.GE2E_loss()

        # Optimizer
        self.optimizer = Optimizer(model_paras, **self.config['hparas'])
        self.verbose(self.optimizer.create_msg())

        # Enable AMP if needed
        self.enable_apex()
        # DistributedDataParallel
        if self.paras.gpu and self.mode == 'train':
            if not self.no_amp:
                self.model = DDP(self.model) 
            else:
                self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu_rank])         
        
        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt()
    
    def exec(self):
        ''' Training End-to-end SV system '''
        self.verbose('Total training steps {}.'.format(human_format(self.max_step)))

        verifi_loss = None
        n_epochs = 0
        self.timer.set()

        while self.step < self.max_step:
            # Renew dataloader to enable random sampling
            pass
            