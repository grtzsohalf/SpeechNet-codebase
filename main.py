#!/usr/bin/env python
# coding: utf-8
import os
import yaml
import torch
import argparse
import numpy as np

import torch.multiprocessing as mp
import torch.distributed as dist

# For reproducibility, comment these may speed up training
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Arguments
parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--config', type=str, help='Path to experiment config.')
parser.add_argument('--name', default=None, type=str, help='Name for logging.')
parser.add_argument('--logdir', default='log/', type=str,
                    help='Logging path.', required=False)
parser.add_argument('--ckpdir', default='ckpt/', type=str,
                    help='Checkpoint path.', required=False)
parser.add_argument('--outdir', default='result/', type=str,
                    help='Decode output path.', required=False)
parser.add_argument('--sampledir', default='samples', type=str,
                    help='Sample data path.')
parser.add_argument('--load', default=None, type=str,
                    help='Load pre-trained model (for training only)', required=False)
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed for reproducable results.', required=False)
parser.add_argument('--cudnn-ctc', action='store_true',
                    help='Switches CTC backend from torch to cudnn')
parser.add_argument('--njobs', default=8, type=int,
                    help='Number of threads for dataloader/decoding.', required=False)
parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
parser.add_argument('--no-pin', action='store_true',
                    help='Disable pin-memory for dataloader')
#parser.add_argument('--test', action='store_true', help='Test the model.')
parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')
parser.add_argument('--eval_init', action='store_true', help='Whether to evaluate at the first iteration')

parser.add_argument('--single_task', action='store_true', help='If single task or not')
parser.add_argument('--pcgrad', action='store_true', help='Whether to use PCGrad')
parser.add_argument('--per_layer', action='store_true', help='Whether to use PCGrad on per layer')
parser.add_argument('--not_collect_grad', action='store_true', help='Whether to collect grad')

parser.add_argument('--asr', action='store_true',
                    help='Option for training ASR.')
parser.add_argument('--sc', action='store_true',
                    help='Option for training Speaker Classification')
parser.add_argument('--se', action='store_true',
                    help='Option for training speech enhancement.')
parser.add_argument('--tts', action='store_true',
                    help='Option for training text-to-speech.')
parser.add_argument('--vcb', action='store_true',
                    help='Option for training Voice Conversion Baseline.')

parser.add_argument('--not_train_vcb', action='store_true',
                    help='Option for NOT training but testing Voice Conversion Baseline.')

parser.add_argument('--test_tts', action='store_true',
                    help='Option for testing tts')       
parser.add_argument('--test_vcb', action='store_true',
                    help='Optibon for testing vcb')      
parser.add_argument('--test_sc', action='store_true',
                    help='Optibon for testing sc')      
parser.add_argument('--test_se', action='store_true',
                    help='Optibon for testing se')      
parser.add_argument('--test_asr', action='store_true',
                    help='Optibon for testing asr')      

parser.add_argument('--save_spk_emb', action='store_true',
                    help='Optibon for saving speaker embeddings')      

# Following features in development.
parser.add_argument('--no_amp', action='store_true', help='Option to disable AMP.')
parser.add_argument('--reserve-gpu', default=0, type=float,
                    help='Option to reserve GPU ram for training.')
parser.add_argument('--jit', action='store_true',
                    help='Option for enabling jit in pytorch. (feature in development)')
parser.add_argument('--local_rank', type=int)
# For multi-gpu
parser.add_argument('--nodes', default=1, type=int, metavar='N')
parser.add_argument('--gpus', default=1, type=int, help='number of gpus per node')
parser.add_argument('--nr', default=0, type=int, help='ranking within the nodes')
parser.add_argument('--master_port', default=6789, type=int, help='port of master')

parser.add_argument('--prof', action='store_true', help='profiling')


###
paras = parser.parse_args()

# For multi-gpu
paras.world_size = paras.gpus * paras.nodes
'''
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(paras.master_port)
'''

if paras.cpu:
    setattr(paras, 'no_amp', True)    
setattr(paras, 'gpu', not paras.cpu)
setattr(paras, 'pin_memory', not paras.no_pin)
setattr(paras, 'verbose', not paras.no_msg)
config = yaml.load(open(paras.config, 'r'), Loader=yaml.FullLoader)

np.random.seed(paras.seed)
torch.manual_seed(paras.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(paras.seed)

# Hack to preserve GPU ram just incase OOM later on server
if paras.gpu and paras.reserve_gpu > 0:
    buff = torch.randn(int(paras.reserve_gpu*1e9//4)).cuda()
    del buff

SolverASR = None
SolverSE = None
SolverTTS = None
SolverSC = None
SolverVCB = None

mode = "train"

# For ASR
if paras.asr:
    if paras.test:
        # Test ASR
        assert paras.load is None, 'Load option is mutually exclusive to --test'
        from bin.asr.test_asr import SolverASR
        mode = 'test'
    else:
        # Train ASR
        from bin.asr.train_asr import SolverASR
        mode = 'train'
    
if paras.sc:
    from bin.sc.solver import SolverSC

# You can specify your own solver for another task, and pass it into GeneralSolver
if paras.se:
    from bin.se.solver import SolverSE
if paras.tts:
    from bin.tts.solver import SolverTTS
if paras.vcb:
    from bin.vc_baseline.solver import SolverVCB

if mode == 'train':
    from bin.general_train_solver import GeneralSolver
else:
    from bin.general_test_solver import GeneralSolver



def solve(config, paras, mode):
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    #rank = paras.nr * paras.gpus + gpu
    rank = paras.local_rank
    print(f'Running DDP on rank {rank}.')
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        #world_size=paras.world_size,
        #rank=rank
    )
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )

    gpu_rank = paras.local_rank
    general_solver = GeneralSolver(SolverASR, SolverSE, SolverTTS, SolverSC, SolverVCB, gpu_rank, paras.world_size, rank, config, paras, mode)

    general_solver.load_data()
    general_solver.set_model()
    general_solver.exec()

    dist.destroy_process_group()

if __name__ == '__main__':
    
    if paras.gpu and mode == 'train':
        solve(config, paras, mode)
    else:
        general_solver = GeneralSolver(SolverASR, SolverSE, SolverTTS, SolverSC, SolverVCB, None, None, None, config, paras, mode)

        general_solver.load_data()
        general_solver.set_model()
        general_solver.exec()
