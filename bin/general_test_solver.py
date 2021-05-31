import torch
import torch.nn as nn
from src.solver import BaseSolver

from src.model import Model


class GeneralSolver(BaseSolver):
    ''' General Solver for training'''

    def __init__(self, SolverASR, SolverLM, gpu_rank, world_size, rank, config, paras, mode):
        super().__init__(config, paras, mode)

        if SolverASR is not None:
            self.solver_asr = SolverASR(gpu_rank, world_size, rank, config, paras, mode)
            self.asr_task = True
        else:
            self.asr_task = False

        if SolverLM is not None:
            self.solver_lm = SolverLM(gpu_rank, world_size, rank, config, paras, mode)
            self.lm_task = True
        else:
            self.lm_task = False

        self.gpu_rank = gpu_rank
        self.world_size = world_size
        self.rank = rank

    def load_data(self):
        ''' ASR: Load data for training/validation, store tokenizer and input/output shape'''
        ''' LM: Load data for training/validation, store tokenizer and input/output shape'''

        def _assert_equal(assert_list):
            all_equal = True
            for i in range(len(assert_list) - 1):
                if assert_list[i] != assert_list[i+1]:
                    all_equal = False
                    break
            return all_equal

        # Check if the shared variables in all tasks are the same
        self.vocab_size = None
        self.tokenizer = None
        self.feat_dim = None
        vocab_size_list = []
        tokenizer_list = []
        feat_dim_list = []
        if self.asr_task:
            self.solver_asr.load_data()
            vocab_size_list.append(self.solver_asr.vocab_size)
            tokenizer_list.append(self.solver_asr.tokenizer)
            feat_dim_list.append(self.solver_asr.feat_dim)
        if self.lm_task:
            self.solver_lm.load_data()
            vocab_size_list.append(self.solver_lm.vocab_size)
            tokenizer_list.append(self.solver_lm.tokenizer)

        for assert_list, assert_message in zip([vocab_size_list, 
                                                tokenizer_list,
                                                feat_dim_list], 
                                               ['vocab_size', 
                                                'tokenizer',
                                                'feat_dim']):
            if len(assert_list) > 0:
                assert _assert_equal([assert_list]) == True, assert_message+' not matched'

        self.vocab_size = vocab_size_list[0]
        self.tokenizer = tokenizer_list[0]
        self.feat_dim = feat_dim_list[0]


    def set_model(self):
        ''' Setup model and optimizer '''
        init_adadelta = False

        ### Model
        if self.paras.gpu and self.mode == 'train' and not self.paras.lm:
            torch.cuda.set_device(self.gpu_rank)
            self.model = Model(self.asr_task, self.lm_task, 
                               self.feat_dim, self.vocab_size, init_adadelta, **self.config['model']).cuda(self.gpu_rank)
        else:
            self.model = Model(self.asr_task, self.lm_task,
                               self.feat_dim, self.vocab_size, init_adadelta, **self.config['model']).to(self.device)

        self.verbose(self.model.create_msg())

        # ASR specific setup
        if self.asr_task:
            self.solver_asr._set_asr_model(self)
        # LM specific setup
        if self.lm_task:
            self.solver_lm._set_lm_model(self)

        # Load target model in eval mode
        self.load_ckpt()

        # ASR decoder setup
        self.solver_asr._set_asr_decoder(self)

    def exec(self):
        ''' Testing a general system '''
        if self.asr_task:
            self.solver_asr._exec_asr(self)
        if self.lm_task:
            self.solver_lm._exec_lm(self)
