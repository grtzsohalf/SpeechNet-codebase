import tqdm
import os
import sys
import torch
import torch.nn as nn
import numpy as np
#from torch.cuda import amp

from src.solver import BaseSolver
from collections import defaultdict
from time import time
import diffdist.functional as distops
from src.model import Model
from src.optim import Optimizer
from bin.asr.asr_data import load_dataset as asr_load_dataset
from src.util import human_format, cal_er, feat_to_fig

from src.pcgrad import PCGrad

from src.transformer.label_smoothing_loss import LabelSmoothingLoss

import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP

import IPython
import pdb

import torch.cuda.nvtx as nvtx




import wandb

class GeneralSolver(BaseSolver):
    ''' General Solver for training'''

    def __init__(self, SolverASR, SolverSE, SolverTTS, SolverSC, SolverVCB, gpu_rank, world_size, rank, config, paras, mode):
        super().__init__(config, paras, mode, rank)
        if SolverASR is not None:
            self.solver_asr = SolverASR(
                gpu_rank, world_size, rank, config, paras, mode, log=self.log)
            self.asr_task = True
        else:
            self.solver_asr = None
            self.asr_task = False

        if SolverSE is not None:
            self.solver_se = SolverSE(
                gpu_rank, world_size, rank, config, paras, mode, self.log)
            self.se_task = True
        else:
            self.se_task = False
            self.solver_se = None

        if SolverTTS is not None:
            self.solver_tts = SolverTTS(
                gpu_rank, world_size, rank, config, paras, mode, self.log)
            self.tts_task = True
        else:
            self.tts_task = False
            self.solver_tts = None 
        
        if SolverSC is not None:
            self.solver_sc = SolverSC(
                gpu_rank, world_size, rank, config, paras, mode, self.log)
            self.sc_task = True
        else:
            self.sc_task = False
            self.solver_sc = None 
        
        if SolverVCB is not None:
            self.solver_vcb = SolverVCB(
                gpu_rank, world_size, rank, config, paras, mode, self.log)
            self.vcb_task = True
            self.not_train_vcb = paras.not_train_vcb
        else:
            self.vcb_task = False
            self.solver_vcb = None
            self.not_train_vcb = False

        self.gpu_rank = gpu_rank
        self.world_size = world_size
        self.rank = rank
        self.config = config
        self.paras = paras
        self.mode = mode

    def load_data(self):
        ''' ASR: Load data for training/validation, store tokenizer and input/output shape'''

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

        start_t = time()
        if self.asr_task:
            self.solver_asr.load_data()
            vocab_size_list.append(self.solver_asr.vocab_size)
            tokenizer_list.append(self.solver_asr.tokenizer)
            feat_dim_list.append(self.solver_asr.feat_dim)
        if self.sc_task:
            feat_dim_list.append(self.solver_sc.input_featdim)
        if self.se_task:
            feat_dim_list.append(self.solver_se.input_featdim)
        if self.sc_task:
            feat_dim_list.append(self.solver_sc.input_featdim)
            #  fetch data
        if self.tts_task:
            feat_dim_list.append(self.solver_tts.input_featdim)
        if self.vcb_task:
            feat_dim_list.append(self.solver_vcb.input_featdim)

        for assert_list, assert_message in zip([vocab_size_list,
                                                tokenizer_list,
                                                feat_dim_list],
                                               ['vocab_size',
                                                'tokenizer',
                                                'feat_dim']):
            if len(assert_list) > 0:
                assert _assert_equal(
                    [assert_list]) == True, assert_message+' not matched'

        self.vocab_size = vocab_size_list[0] if len(
            vocab_size_list) > 0 else 1
        self.tokenizer = tokenizer_list[0] if len(tokenizer_list) > 0 else None
        self.feat_dim = feat_dim_list[0] if len(feat_dim_list) > 0 else 1
        print(f'Time of loading data: {time() - start_t}')

    def set_model(self):
        ''' Setup model and optimizer '''
        init_adadelta = self.config['hparas']['optimizer'] == 'Adadelta'

        # Model
        if self.paras.gpu and self.mode == 'train':
            torch.cuda.set_device(self.gpu_rank)
            self.process_group = torch.distributed.new_group()
            self.model = Model(self.paras, self.config, self.mode, self.asr_task, self.se_task, self.tts_task, self.sc_task, self.vcb_task, 
                               self.feat_dim, self.vocab_size, init_adadelta, \
                               **self.config['model'],
                               solver_asr=self.solver_asr, solver_se=self.solver_se, solver_tts=self.solver_tts,
                               solver_sc=self.solver_sc, solver_vcb=self.solver_vcb).cuda(self.gpu_rank)
        else:
            self.model = Model(self.paras, self.config, self.mode, self.asr_task, self.se_task, self.tts_task,
                               self.sc_task, self.vcb_task, self.feat_dim, self.vocab_size, init_adadelta, **self.config['model'], 
                               solver_asr=self.solver_asr, solver_se=self.solver_se, solver_tts=self.solver_tts,
                               solver_sc=self.solver_sc, solver_vcb=self.solver_vcb).to(self.device)

        self.verbose(self.model.create_msg())

        # Optimizer
        '''
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                print(n)
        # print([n for n, p in self.model.named_parameters() if p.requires_grad])
        exit()
        '''

        no_decay = ["bias", "norm_ff.weight", "norm_mha.weight", "norm_conv.weight", "norm_final.weight", "log_sigma"]

        '''
        model_paras = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        '''
        model_paras = {}
        self.optimizer_dict = {}
        module_list = ['audio_transform', 'audio_encoder', 'audio_decoder', 'text_encoder', 'text_decoder', 'prosody_predictor',
                       'asr', 'se', 'sc', 'tts', 'vcb'] 
        if not self.paras.single_task:
            module_list += ['log_sigma_asr', 'log_sigma_se', 'log_sigma_sc', 'log_sigma_tts', 'log_sigma_vcb']

        for module_name in module_list:
            model_paras[module_name] = []
            if len([p for n, p in self.model.named_parameters() \
                    if any(nd in n for nd in no_decay) and (module_name == n[:len(module_name)])]) > 0:
                model_paras[module_name] += [
                    {
                        "params": [p for n, p in self.model.named_parameters() \
                                   if any(nd in n for nd in no_decay) and (module_name == n[:len(module_name)])], 
                        "weight_decay": 0.0
                    },
                ]
            if len([p for n, p in self.model.named_parameters() \
                    if (not any(nd in n for nd in no_decay)) and (module_name == n[:len(module_name)])]) > 0:
                model_paras[module_name] += [
                    {
                        "params": [p for n, p in self.model.named_parameters() \
                                   if (not any(nd in n for nd in no_decay)) and (module_name == n[:len(module_name)])],
                        "weight_decay": 0.01,
                    },
                ]

            if len(model_paras[module_name]) != 0:
                module_opt = Optimizer(model_paras[module_name], **self.config['hparas'])
                if self.paras.pcgrad or not self.paras.not_collect_grad:
                    module_opt = PCGrad(module_opt)
                self.verbose(module_opt.create_msg())
                self.optimizer_dict[module_name] = module_opt

        # ASR specific setup
        if self.asr_task:
            model_paras = self.solver_asr._set_asr_model(self, model_paras)

        # Enable AMP if needed
        self.enable_apex()
        # DistributedDataParallel
        if self.paras.gpu and self.mode == 'train':
            if not self.no_amp:
                self.model = DDP(self.model)
            else:
                #self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model, self.process_group)
                self.model = nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[self.gpu_rank], find_unused_parameters=True)
                
        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt()

    def _iter_data(self, iter_item, data_loader, exhausted_flag):
        try:
            data = next(iter_item)
            # if len(data[0]) < self.world_size:
                # iter_item = iter(data_loader)
                # exhausted_flag = True
                # data = next(iter_item)
        except StopIteration:
            iter_item = iter(data_loader)
            exhausted_flag = True
            data = next(iter_item)
        return iter_item, data, exhausted_flag

    def exec(self):
        ''' Training a general system '''
        self.verbose('Total training steps {}.'.format(
            human_format(self.max_step)))

        # if self.asr_task:
        # ctc_loss, att_loss, emb_loss = None, None, None

        pbar = tqdm.tqdm(total=self.max_step, dynamic_ncols=True)
        pbar.n = self.step
        n_epochs = 0
        self.timer.set()

        if self.paras.test_tts or self.paras.test_vcb or self.paras.test_sc or self.paras.test_se or self.paras.test_asr or self.paras.save_spk_emb:
            for _ in range(1):
                self.validate()
            exit()

        while self.step < self.max_step:
            # To enable shuffling at each epoch
            if self.solver_asr and self.solver_asr.tr_sampler:
                self.solver_asr.tr_sampler.set_epoch(n_epochs)

            if self.asr_task:
                self.solver_asr._renew_asr_dataset(self, n_epochs)
            if self.se_task:
                self.solver_se.set_trainloader()
                self.solver_se.tr_sampler.set_epoch(n_epochs)
            if self.tts_task:
                self.solver_tts.set_trainloader()
                self.solver_tts.tr_sampler.set_epoch(n_epochs)
            if self.sc_task:
                self.solver_sc.set_train_dataloader()
                self.solver_sc.tr_sampler.set_epoch(n_epochs)
            if self.vcb_task and not self.not_train_vcb:
                self.solver_vcb.set_trainloader()
                self.solver_vcb.tr_sampler.set_epoch(n_epochs)

            asr_data_exhausted = False
            se_data_exhausted = False
            tts_data_exhausted = False
            sc_data_exhausted = False
            vcb_data_exhausted = False

            if self.asr_task:
                asr_tr_set_iter = iter(self.solver_asr.tr_set)
            if self.se_task:
                se_tr_set_iter = iter(self.solver_se.trainloader)
            if self.tts_task:
                tts_tr_set_iter = iter(self.solver_tts.trainloader)
            if self.sc_task:
                sc_tr_set_iter = iter(self.solver_sc.train_dataloader)
            if self.vcb_task and not self.not_train_vcb:
                vcb_tr_set_iter = iter(self.solver_vcb.trainloader)

            start_time = time()
            while not asr_data_exhausted or not se_data_exhausted or not tts_data_exhausted \
                or not sc_data_exhausted or not vcb_data_exhausted:

                if self.rank == 0:
                    diff_time = time()-start_time
                    start_time = time()

                nvtx.range_push(f'Step {self.step}')

                # Get data
                nvtx.range_push('Get data')
                if self.asr_task:
                    asr_tr_set_iter, asr_data, asr_data_exhausted = \
                        self._iter_data(
                            asr_tr_set_iter, self.solver_asr.tr_set, asr_data_exhausted)
                else:
                    asr_data = None
                    asr_data_exhausted = True

                if self.se_task:
                    se_tr_set_iter, se_data, se_data_exhausted = \
                        self._iter_data(
                            se_tr_set_iter, self.solver_se.trainloader, se_data_exhausted)
                else:
                    se_data = None
                    se_data_exhausted = True
                
                if self.tts_task:
                    tts_tr_set_iter, tts_data, tts_data_exhausted = \
                        self._iter_data(
                            tts_tr_set_iter, self.solver_tts.trainloader, tts_data_exhausted)
                else:
                    tts_data = None
                    tts_data_exhausted = True
                
                if self.sc_task:
                    sc_tr_set_iter, sc_data, sc_data_exhausted = \
                        self._iter_data(sc_tr_set_iter, self.solver_sc.train_dataloader, sc_data_exhausted)
                else:
                    sc_data = None
                    sc_data_exhausted = True
                
                if self.vcb_task and not self.not_train_vcb:
                    vcb_tr_set_iter, vcb_data, vcb_data_exhausted = \
                        self._iter_data(vcb_tr_set_iter, self.solver_vcb.trainloader, vcb_data_exhausted)
                    '''
                    vcb_skip = False
                    if torch.max(vcb_data[1]) + torch.max(vcb_data[3]) > 160000:
                        print('vcb_data too long at step: {}'.format(self.step))
                        vcb_skip = True
                    '''
                else:
                    vcb_data = None
                    vcb_data_exhausted = True

                nvtx.range_pop()


                if self.rank == 0:
                    iter_data_time = time()-start_time

                if (asr_data_exhausted and se_data_exhausted and tts_data_exhausted and sc_data_exhausted and vcb_data_exhausted):
                    break
                '''
                if self.vcb_task:
                    if vcb_skip and asr_data_exhausted and lm_data_exhausted and se_data_exhausted and sv_data_exhausted and ss_data_exhausted and ssw_data_exhausted and tts_data_exhausted and sc_data_exhausted:
                        continue
                '''

                for name, opt in self.optimizer_dict.items():
                    opt.zero_grad()
                    if opt.lr_scheduler is not None:
                        cur_lr = opt.lr_scheduler.get_last_lr()[0]
                tf_rate = 1.

                # total_loss = 0

                # Fetch data
                # fetched_asr_data: feat, feat_len, txt, txt_len
                # fetched_lm_data: txt, txt_len
                # fetched_asr_data, fetched_lm_data = self.fetch_data(
                    # asr_data, lm_data)
                self.timer.cnt('rd')

                # Forward
                asr_inputs = None
                if self.asr_task:
                    # asr_inputs = {'audio_feature': fetched_asr_data['feat'],
                                # 'feature_len': fetched_asr_data['feat_len'],
                                # 'decode_step': max(fetched_asr_data['txt_len']),
                                # 'tf_rate': tf_rate,
                                # 'teacher': fetched_asr_data['txt'],
                                # 'emb_decoder': None,
                                # 'get_dec_state': self.solver_asr.emb_reg,
                                # 'txt_len': fetched_asr_data['txt_len']}
                    asr_inputs = {'asr_data': asr_data,
                                'tf_rate': tf_rate,
                                'emb_decoder': None,
                                'get_dec_state': self.solver_asr.emb_reg,
                                'mode': 'train'}
                    # ctc_output, encode_len, att_output, att_align, dec_state, pred, _ = \
                    # self.model(feat, feat_len, max(txt_len), tf_rate, txt, None, self.solver_asr.emb_reg, txt_len,
                    # None, None, None)

                #with amp.autocast(enabled=(not self.no_amp)):
                # asr_outputs: ctc_output, encode_len, att_output, att_align, dec_state
                # lm_outputs: pred, hidden
                nvtx.range_push('Model forward')

                loss = 0.
                if self.paras.pcgrad or not self.paras.not_collect_grad:
                    grad_dict = {}
                    shape_dict = {}
                    has_grad_dict = {}
                    for name, _ in self.optimizer_dict.items():
                        grad_dict[name] = []
                        shape_dict[name] = []
                        has_grad_dict[name] = []
                else:
                    grad_dict = None
                    shape_dict = None
                    has_grad_dict = None

                # Forward only once 
                if not self.paras.pcgrad and self.paras.not_collect_grad:
                    asr_outputs, se_outputs, sc_outputs, tts_outputs, vcb_outputs, loss = self.model(
                        asr_inputs, se_data, tts_data, sc_data, vcb_data,
                        self.step, cur_lr, self.PROGRESS_STEP, 'train')

                else:
                    # asr forward
                    if self.asr_task and asr_inputs is not None:
                        asr_outputs, _, _, _, _, scaled_asr_loss = self.model(
                                asr_inputs, None, None, None, None, 
                                self.step, cur_lr, self.PROGRESS_STEP, 'train')
                        loss = loss + scaled_asr_loss
                        asr_module_list = ['audio_transform', 'audio_encoder', 'text_decoder', 'asr', 'prosody_predictor']
                        if not self.paras.single_task:
                            asr_module_list.append('log_sigma_asr')
                        for module_name in asr_module_list:
                            self.optimizer_dict[module_name].zero_grad()
                        scaled_asr_loss.backward()#(retain_graph=True)
                        for module_name in asr_module_list:
                            grad, shape, has_grad = self.optimizer_dict[module_name]._retrieve_grad()
                            grad_dict[module_name].append(grad)
                            shape_dict[module_name].append(shape)
                            has_grad_dict[module_name].append(has_grad)
                            # optimizer_dict[module_name].zero_grad(set_to_none=False)
                            # optimizer_dict[module_name].step()

                    # se forward
                    if self.se_task and se_data:
                        _, se_outputs, _, _, _, scaled_se_loss = self.model(
                                None, se_data, None, None, None, 
                                self.step, cur_lr, self.PROGRESS_STEP, 'train')
                        loss = loss + scaled_se_loss
                        se_module_list = ['audio_transform', 'audio_encoder', 'audio_decoder', 'se', 'prosody_predictor']
                        if not self.paras.single_task:
                            se_module_list.append('log_sigma_se')
                        for module_name in se_module_list:
                            self.optimizer_dict[module_name].zero_grad()
                        scaled_se_loss.backward()#(retain_graph=True)
                        for module_name in se_module_list:
                            grad, shape, has_grad = self.optimizer_dict[module_name]._retrieve_grad()
                            grad_dict[module_name].append(grad)
                            shape_dict[module_name].append(shape)
                            has_grad_dict[module_name].append(has_grad)
                            # optimizer_dict[module_name].zero_grad(set_to_none=False)
                            # optimizer_dict[module_name].step()

                    # sc forward
                    if self.sc_task and sc_data:
                        _, _, sc_outputs, _, _, scaled_sc_loss = self.model(
                                None, None, None, sc_data, None, 
                                self.step, cur_lr, self.PROGRESS_STEP, 'train')
                        loss = loss + scaled_sc_loss
                        sc_module_list = ['audio_transform', 'audio_encoder', 'sc']
                        if not self.paras.single_task:
                            sc_module_list.append('log_sigma_sc')
                        for module_name in sc_module_list:
                            self.optimizer_dict[module_name].zero_grad()
                        scaled_sc_loss.backward()#(retain_graph=True)
                        for module_name in sc_module_list:
                            grad, shape, has_grad = self.optimizer_dict[module_name]._retrieve_grad()
                            grad_dict[module_name].append(grad)
                            shape_dict[module_name].append(shape)
                            has_grad_dict[module_name].append(has_grad)
                            # optimizer_dict[module_name].zero_grad(set_to_none=False)
                            # optimizer_dict[module_name].step()

                    # tts forward
                    if self.tts_task and tts_data:
                        _, _, _, tts_outputs, _, scaled_tts_total_loss = self.model(
                                None, None, tts_data, None, None, 
                                self.step, cur_lr, self.PROGRESS_STEP, 'train')
                        loss = loss + scaled_tts_total_loss
                        tts_module_list = ['audio_transform', 'audio_decoder', 'text_encoder', 'tts', 'prosody_predictor']
                        if not self.paras.single_task:
                            tts_module_list.append('log_sigma_tts')
                        for module_name in tts_module_list:
                            self.optimizer_dict[module_name].zero_grad()
                        scaled_tts_total_loss.backward()#(retain_graph=True)
                        for module_name in tts_module_list:
                            grad, shape, has_grad = self.optimizer_dict[module_name]._retrieve_grad()
                            grad_dict[module_name].append(grad)
                            shape_dict[module_name].append(shape)
                            has_grad_dict[module_name].append(has_grad)
                            # optimizer_dict[module_name].zero_grad(set_to_none=False)
                            # optimizer_dict[module_name].step()

                    # vcb forward
                    if self.vcb_task and vcb_data:
                        _, _, _, _, vcb_outputs, scaled_vcb_total_loss = self.model(
                                None, None, None, None, vcb_data, 
                                self.step, cur_lr, self.PROGRESS_STEP, 'train')
                        loss = loss + scaled_vcb_total_loss
                        vcb_module_list = ['audio_transform', 'audio_encoder', 'audio_decoder', 'vcb', 'prosody_predictor']
                        if not self.paras.single_task:
                            vcb_module_list.append('log_sigma_vcb')
                        for module_name in vcb_module_list:
                            self.optimizer_dict[module_name].zero_grad()
                        scaled_vcb_total_loss.backward()#(retain_graph=True)
                        for module_name in vcb_module_list:
                            grad, shape, has_grad = self.optimizer_dict[module_name]._retrieve_grad()
                            grad_dict[module_name].append(grad)
                            shape_dict[module_name].append(shape)
                            has_grad_dict[module_name].append(has_grad)
                            # optimizer_dict[module_name].zero_grad(set_to_none=False)
                            # optimizer_dict[module_name].step()

                    
                    '''
                    if self.rank == 0:
                        print(grad_dict['audio_encoder'])
                        print(shape_dict['audio_encoder'])
                        print(has_grad_dict['audio_encoder'])
                        exit()
                    '''
                
                nvtx.range_pop()

                if self.rank == 0:
                    forward_time = time()-start_time

                if self.rank == 0:
                    loss_time = time()-start_time

                self.timer.cnt('fw')
                
                if self.paras.test_tts and self.step%10==0:
                    tts_dev_loss, tts_dev_mel_loss, tts_dev_d_loss, tts_dev_speaker_loss, tts_sample_mel, tts_target_mel, tts_sample_wav, tts_target_wav = self.solver_tts.compute_metrics(**tts_outputs, split='dev', synthesis=True)
                    self.solver_tts.logging(f"tts_sample_mel_train_{self.step}", tts_sample_mel, self.step, mode='image')
                    self.solver_tts.logging(f"tts_target_mel_train_{self.step}", tts_target_mel, self.step, mode='image')
                    self.solver_tts.logging(f"TTS_sample_wav_train{self.step}", tts_sample_wav, self.step, mode='wav')
                    self.solver_tts.logging(f"TTS_target_wav_train{self.step}", tts_target_wav, self.step, mode='wav')

                if self.paras.test_vcb:
                    self.solver_vcb.compute_metrics(**vcb_outputs, split='dev')
                    if self.step % 10 == 0:
                        self.solver_vcb.log_evaluation(step=self.step, split='dev')

                # logging
                if self.rank == 0 and self.step % self.config['model']['log_step'] == 0:
                    self.log.add_scalar('total_loss', loss.item(), global_step=self.step)
                    self.log.add_scalar('asr_log_sigma', self.model.module.log_sigma_asr[0].item(), global_step=self.step)
                    self.log.add_scalar('se_log_sigma', self.model.module.log_sigma_se[0].item(), global_step=self.step)
                    self.log.add_scalar('sc_log_sigma', self.model.module.log_sigma_sc[0].item(), global_step=self.step)
                    self.log.add_scalar('tts_log_sigma', self.model.module.log_sigma_tts[0].item(), global_step=self.step)
                    self.log.add_scalar('vcb_log_sigma', self.model.module.log_sigma_vcb[0].item(), global_step=self.step)

                # Backprop
                if not self.paras.test_tts and not self.paras.test_vcb and not self.paras.test_sc:
                    nvtx.range_push('Backward')
                    grad_norm, total_norm, grad_conflict_dict = self.backward(
                        loss, grad_dict, shape_dict, has_grad_dict, criterion=None, vector=None)
                    nvtx.range_pop()
                    #self.step += 1
                    if self.rank == 0:
                        backward_time = time()-start_time
                self.step += 1
                pbar.update(1)

                # Post-step : update tf_rate/lr_rate
                # tf_rate: teacher forcing rate
                # cur_lr: current learning rate
                nvtx.range_push('Opt post_step')
                for name, opt in self.optimizer_dict.items():
                    tf_rate, cur_lr = opt.post_step(self.step)
                nvtx.range_pop()
                if self.rank == 0:
                    opt_post_time = time()-start_time

                # Calculate grad conflicting
                if self.rank == 0 and self.step % self.config['model']['log_step'] == 0:
                    print(f'Total grad norm: {total_norm}')
                    self.log.add_scalar(f'total_grad_norm', total_norm, global_step=self.step)
                    with open(os.path.join(self.outdir, 'grad.txt'), 'a') as f_grad:
                        f_grad.write(f'Step {self.step}: Grad norm = {total_norm}\n')
                        all_total_counts = 0
                        all_conflict_counts = 0
                        for name, count_tuple in grad_conflict_dict.items():
                            if 'encoder' in name or 'decoder' in name:
                                all_total_counts += count_tuple[0]
                                all_conflict_counts += count_tuple[1]
                                if count_tuple[0] == 0:
                                    count_ratio = 0
                                else:
                                    count_ratio = count_tuple[1] / count_tuple[0]
                                f_grad.write(f'{name}: total_counts = {count_tuple[0]}, conflict_counts = {count_tuple[1]}, conflict_ratio = {count_ratio}\n')
                                self.log.add_scalar(f'grad_total_counts/{name}', count_tuple[0], global_step=self.step)
                                self.log.add_scalar(f'grad_conflict_counts/{name}', count_tuple[1], global_step=self.step)
                                self.log.add_scalar(f'grad_conflict_ratio/{name}', count_ratio, global_step=self.step)
                        if all_total_counts == 0:
                            all_conflict_ratio = 0
                        else:
                            all_conflict_ratio = all_conflict_counts / all_total_counts
                        self.log.add_scalar(f'grad_total_counts/all', all_total_counts, global_step=self.step)
                        self.log.add_scalar(f'grad_conflict_counts/all', all_conflict_counts, global_step=self.step)
                        self.log.add_scalar(f'grad_conflict_ratio/all', all_conflict_ratio, global_step=self.step)
                        f_grad.write(f'all: total_counts = {all_total_counts}, conflict_counts = {all_conflict_counts}, conflict_ratio = {all_conflict_ratio}\n')


                # Validation
                # if (self.rank == 0 or self.rank is None) and ((self.step == 1 and self.paras.eval_init) or (self.step % self.valid_step == 0)):
                nvtx.range_push('Validate')
                if ((self.step == 1 and self.paras.eval_init) or (self.step % self.valid_step == 0)):
                    st = time()

                    # with amp.autocast(enabled=False):
                    self.validate()
                    spend = time() - st
                    print(f'Validation spends: {spend} sec')
                nvtx.range_pop()
                if self.rank == 0:
                    validation_time = time()-start_time

                # End of step
                # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
                '''
                nvtx.range_push('Empty cache')
                torch.cuda.empty_cache()
                nvtx.range_pop()
                '''

                '''
                if self.rank == 0:
                    print(f'Diff with last step: {diff_time}')
                    print(f'Start of step: {start_time}')
                    print(f'End of iter_data: {iter_data_time}')
                    print(f'End of opt_post: {opt_post_time}')
                    print(f'End of forward: {forward_time}')
                    print(f'End of loss computation: {loss_time}')
                    print(f'End of backward: {backward_time}')
                    print(f'End of validation: {validation_time}')
                    print(f'End of dist barrier: {dist_barrier_time}')
                    print(self.timer.time_table)
                    print(self.timer.show())
                    print('===============')
                    print('===============')
                '''
                self.timer.clear()
                self.timer.set()
                if self.step > self.max_step:
                    break

                nvtx.range_pop()

            if self.paras.test_tts or self.paras.test_vcb or self.paras.test_sc:
                break
            n_epochs += 1
        #self.log.close()
        pbar.close()

    def validate(self):
        # Eval mode
        print(f'Validation at step {self.step}...')
        self.model.eval()

        if self.asr_task:
            if self.solver_asr.emb_decoder is not None:
                self.solver_asr.emb_decoder.eval()
            asr_dev_wer = {'att': [], 'ctc': []}
            asr_spk_emb_list = []
            asr_spk_list = []
        if self.se_task:
            self.solver_se.init_evaluation(split='dev')
            se_spk_emb_list = []
            se_spk_list = []
        
        if self.sc_task:
            if self.solver_sc.rank == 0:
                self.sc_dev_tensor_list = torch.tensor([], dtype=torch.int32).cuda(self.gpu_rank)
                self.sc_acc_list = []
                sc_spk_emb_list = []
                sc_spk_list = []
        
        if self.tts_task:
            self.tts_dev_loss_list = []
            self.tts_dev_mel_loss_list = []
            self.tts_dev_d_loss_list = []
            self.tts_dev_speaker_loss_list = []
            self.tts_dev_prosody_loss_list = []
            self.tts_sample_mel_list = []
            self.tts_target_mel_list = []
            self.tts_sample_wav_list = []
            self.tts_target_wav_list = []
            tts_spk_emb_list = []
            tts_spk_list = []

        if self.vcb_task:
            self.vcb_dev_loss_list = [[], [], []]
            self.solver_vcb.init_evaluation(split='dev')
            vcb_spk_emb_list = []
            vcb_spk_list = []

        asr_data_exhausted = False
        se_data_exhausted = False
        tts_data_exhausted = False
        sc_data_exhausted = False
        vcb_data_exhausted = False

        if self.asr_task:
            asr_dv_set_iter = iter(self.solver_asr.dv_set)
        if self.se_task:
            if not (self.paras.test_se):
                se_dv_set_iter = iter(self.solver_se.devloader)
            else:
                se_dv_set_iter = iter(self.solver_se.testloader)
        if self.tts_task:
            tts_dv_set_iter = iter(self.solver_tts.devloader)
        if self.sc_task:
            sc_dv_set_iter = iter(self.solver_sc.dev_dataloader)
        if self.vcb_task:
            vcb_dv_set_iter = iter(self.solver_vcb.devloader)
            
        count_asr = -1
        count_se = -1
        count_tts = -1
        count_sc = -1
        count_vcb = -1

        while not asr_data_exhausted or not se_data_exhausted \
        or not tts_data_exhausted or not sc_data_exhausted or not vcb_data_exhausted:

            # Get data

            if self.asr_task:
                asr_dv_set_iter, asr_data, asr_data_exhausted = \
                    self._iter_data(asr_dv_set_iter,
                                    self.solver_asr.dv_set, asr_data_exhausted)
            else:
                asr_data = None
                asr_data_exhausted = True

            if self.se_task and not se_data_exhausted:
                if not (self.paras.test_se):
                    se_dv_set_iter, se_data, se_data_exhausted = \
                        self._iter_data(
                            se_dv_set_iter, self.solver_se.devloader, se_data_exhausted)
                else:
                    se_dv_set_iter, se_data, se_data_exhausted = \
                        self._iter_data(
                            se_dv_set_iter, self.solver_se.testloader, se_data_exhausted)
            else:
                se_data = None
                se_data_exhausted = True
            
            if self.sc_task and not sc_data_exhausted:
                sc_dv_set_iter, sc_data, sc_data_exhausted = \
                    self._iter_data(sc_dv_set_iter, self.solver_sc.dev_dataloader, sc_data_exhausted)
                
            else:
                sc_data = None
                sc_data_exhausted = True

            if self.tts_task and not tts_data_exhausted:
                tts_dv_set_iter, tts_data, tts_data_exhausted = \
                    self._iter_data(tts_dv_set_iter, self.solver_tts.devloader, tts_data_exhausted)
            else:
                tts_data = None
                tts_data_exhausted = True
            
            if self.vcb_task and not vcb_data_exhausted:
                vcb_dv_set_iter, vcb_data, vcb_data_exhausted = \
                    self._iter_data(vcb_dv_set_iter, self.solver_vcb.devloader, vcb_data_exhausted)
                '''
                vcb_skip = False
                if torch.max(vcb_data[1]) > 90000:
                    print('vcb_data too long at step:{}'.format(self.step))
                    vcb_skip = True
                '''

            else:
                vcb_data = None
                vcb_data_exhausted = True

            if (asr_data_exhausted and se_data_exhausted \
                and tts_data_exhausted and sc_data_exhausted and vcb_data_exhausted):
                break

            else:
                if not asr_data_exhausted:
                    count_asr += 1
                if not se_data_exhausted:
                    count_se += 1
                if not tts_data_exhausted:
                    count_tts += 1
                if not sc_data_exhausted:
                    count_sc +=1
                if not vcb_data_exhausted:
                    count_vcb += 1
            '''
            if self.vcb_task:
                if vcb_skip and asr_data_exhausted and lm_data_exhausted and se_data_exhausted and sv_data_exhausted and ss_data_exhausted and ssw_data_exhausted and tts_data_exhausted and sc_data_exhausted:
                    continue
            '''
            verbose_string = ""
            if self.asr_task:
                verbose_string += 'ASR - {}/{}'.format(count_asr+1, len(self.solver_asr.dv_set)) + "\t"
            if self.se_task:
                if not self.paras.test_se:
                    verbose_string += 'SE - {}/{}'.format(count_se+1, len(self.solver_se.devloader)) + "\t"
                else:
                    verbose_string += 'SE - {}/{}'.format(count_se+1, len(self.solver_se.testloader)) + "\t"

            if self.tts_task:
                verbose_string += 'TTS - {}/{}'.format(count_tts+1, len(self.solver_tts.devloader)) + "\t"

            if self.sc_task:
                verbose_string += 'SC - {}/{}'.format(count_sc+1, len(self.solver_sc.dev_dataloader)) + "\t"
            
            if self.vcb_task:
                verbose_string += 'VCB - {}/{}'.format(count_vcb+1, len(self.solver_vcb.devloader))

            self.progress(verbose_string)        
            if self.rank == 0:
                self.progress(verbose_string)        
            # Fetch data
            # fetched_asr_data, fetched_lm_data = self.fetch_data(
                # asr_data, lm_data)
            # if self.asr_task:
            # feat, feat_len, txt, txt_len = fetched_asr_data
            # if self.lm_task:
            # txt, txt_len = fetched_lm_data

            # Forward model
            with torch.no_grad():
                asr_inputs = None
                if self.asr_task:
                    # asr_inputs = {'audio_feature': fetched_asr_data['feat'],
                                  # 'feature_len': fetched_asr_data['feat_len'],
                                  # 'decode_step': int(max(fetched_asr_data['txt_len'])*self.solver_asr.DEV_STEP_RATIO),
                                  # 'tf_rate': None,
                                  # 'teacher': None,
                                  # 'emb_decoder': self.solver_asr.emb_decoder,
                                  # 'get_dec_state': None,
                                  # 'txt_len': None}
                    asr_inputs = {'asr_data': asr_data,
                                  'tf_rate': None,
                                  'emb_decoder': self.solver_asr.emb_decoder,
                                  'get_dec_state': None,
                                  'mode': 'test'}
                    # ctc_output, encode_len, att_output, att_align, dec_state, pred, _ = \
                    # self.model(feat, feat_len, int(max(txt_len)*self.solver_asr.DEV_STEP_RATIO),
                    # None, None, self.solver_asr.emb_decoder, None, None,
                    # None, None, None)
                

                # asr_outputs: ctc_output, encode_len, att_output, att_align, dec_state
                # lm_outputs: pred, hidden
                '''
                asr_outputs, lm_outputs, se_outputs, sv_outputs, ss_outputs, ssw_outputs, tts_outputs, sc_outputs, vcb_outputs, _, \
                    _, _, _, _, _ = self.model(
                    asr_inputs, lm_inputs, se_data, sv_data, ss_data, ssw_data, tts_data, sc_data, vcb_data, self.step, None, None, 'valid')
                '''

                asr_outputs, se_outputs, sc_outputs, tts_outputs, vcb_outputs, loss = self.model(
                        asr_inputs, se_data, tts_data, sc_data, vcb_data, 
                        self.step, None, None, 'val')

                if self.step == 317:
                    print( (vcb_outputs == None ) )

            if self.asr_task and not asr_data_exhausted:
                asr_dev_wer = self.solver_asr._validate_asr(self, asr_outputs['txt'],
                                                            asr_outputs['att_output'],
                                                            asr_outputs['ctc_output'],
                                                            asr_dev_wer, count_asr,
                                                            asr_outputs['att_align'], self.step)
                # print(asr_dev_wer)
                att_dev_wer = torch.tensor(asr_dev_wer['att']).cuda(self.gpu_rank)
                ctc_dev_wer = torch.tensor(asr_dev_wer['ctc']).cuda(self.gpu_rank)
                # gathered_att_dev_wer = [torch.tensor([1.]*len(att_dev_wer)).cuda(self.gpu_rank)]*self.world_size
                # gathered_ctc_dev_wer = [torch.tensor([1.]*len(ctc_dev_wer)).cuda(self.gpu_rank)]*self.world_size
                # dist.all_gather(gathered_att_dev_wer, att_dev_wer)
                # dist.all_gather(gathered_ctc_dev_wer, ctc_dev_wer)
                # print('gathered_att: ', gathered_att_dev_wer, self.rank)
                # print('gathered_ctc: ', gathered_ctc_dev_wer, self.rank)
                dist.all_reduce(att_dev_wer)
                dist.all_reduce(ctc_dev_wer)
                att_dev_wer /= self.world_size
                ctc_dev_wer /= self.world_size
                # print('reduced_att: ', att_dev_wer, self.rank)
                # print('reduced_ctc: ', ctc_dev_wer, self.rank)
                if self.paras.save_spk_emb:
                    asr_spk_emb_list += asr_outputs['spk_emb']
                    asr_spk_list += asr_outputs['spk']

            if self.se_task and not se_data_exhausted:
                self.solver_se.compute_metrics(**se_outputs, split='dev')
                if self.paras.save_spk_emb:
                    se_spk_emb_list += se_outputs['spk_emb']
                    se_spk_list += se_outputs['spk']
            
            if self.sc_task and not sc_data_exhausted:
                
                sc_dev_list = self.solver_sc.compute_metrics(**sc_outputs, split='dev')
                sc_dev_tensors=torch.tensor(sc_dev_list).cuda(self.gpu_rank)
                sc_dev_tensors_list = [torch.zeros_like(sc_dev_tensors) for _ in range(self.world_size)]
                torch.distributed.all_gather(sc_dev_tensors_list, sc_dev_tensors)

                if self.solver_sc.rank == 0:
                    for sc_dev in sc_dev_tensors_list:
                        self.sc_dev_tensor_list = torch.cat((self.sc_dev_tensor_list, sc_dev), dim=0)

                if self.paras.save_spk_emb:
                    sc_spk_emb_list += sc_outputs['spk_emb']
                    sc_spk_list += sc_outputs['spk']
            
            if self.tts_task and not tts_data_exhausted:
                if len(self.tts_sample_wav_list) < 10:
                    tts_dev_loss, tts_dev_mel_loss, tts_dev_d_loss, tts_dev_speaker_loss, tts_dev_prosody_loss, \
                        tts_sample_mel, tts_target_mel, tts_sample_wav, tts_target_wav = \
                        self.solver_tts.compute_metrics(**tts_outputs, split='dev', synthesis=True)
                else:
                    tts_dev_loss, tts_dev_mel_loss, tts_dev_d_loss, tts_dev_speaker_loss, tts_dev_prosody_loss, \
                        tts_sample_mel, tts_target_mel, tts_sample_wav, tts_target_wav = \
                        self.solver_tts.compute_metrics(**tts_outputs, split='dev', synthesis=False)
                
                if tts_sample_mel is not None:
                    self.tts_sample_mel_list.append(tts_sample_mel)
                if tts_target_mel is not None:
                    self.tts_target_mel_list.append(tts_target_mel)
                if tts_sample_wav is not None:
                    self.tts_sample_wav_list.append(tts_sample_wav)
                if tts_target_wav is not None:
                    self.tts_target_wav_list.append(tts_target_wav)

                self.tts_dev_loss_list.append(tts_dev_loss)
                self.tts_dev_mel_loss_list.append(tts_dev_mel_loss)
                self.tts_dev_d_loss_list.append(tts_dev_d_loss)
                self.tts_dev_speaker_loss_list.append(tts_dev_speaker_loss)
                self.tts_dev_prosody_loss_list.append(tts_dev_prosody_loss)
                #assert len(self.tts_sample_wav_list) == 1

                if self.paras.save_spk_emb:
                    tts_spk_emb_list += tts_outputs['spk_emb']
                    tts_spk_list += tts_outputs['spk']
            
            #if self.vcb_task and not vcb_data_exhausted and vcb_skip == False:
            if self.vcb_task and not vcb_data_exhausted:
                if self.solver_vcb.rank == 0:
                    self.solver_vcb.compute_metrics(**vcb_outputs, split='dev')
                dev_loss_recon, dev_loss_convert, dev_loss_prosody = self.solver_vcb.compute_loss(**vcb_outputs, split='dev')
                dev_loss_recon_list = [torch.zeros_like(dev_loss_recon) for i in range(self.world_size)]
                dev_loss_convert_list = [torch.zeros_like(dev_loss_convert) for i in range(self.world_size)]
                dev_loss_prosody_list = [torch.zeros_like(dev_loss_prosody) for i in range(self.world_size)]
                dist.all_gather(dev_loss_recon_list, dev_loss_recon)
                dist.all_gather(dev_loss_convert_list, dev_loss_convert)
                dist.all_gather(dev_loss_prosody_list, dev_loss_prosody)
                self.vcb_dev_loss_list[0].extend(dev_loss_recon_list)
                self.vcb_dev_loss_list[1].extend(dev_loss_convert_list)
                self.vcb_dev_loss_list[2].extend(dev_loss_prosody_list)

                if self.paras.save_spk_emb:
                    vcb_spk_emb_list += vcb_outputs['spk_emb']
                    vcb_spk_list += vcb_outputs['spk']

        # Ckpt if performance improves
        if self.rank == 0:
            metric_dict = {}

        if self.asr_task:
            if self.rank == 0:
                #self.solver_asr._check_asr_improvement(self, asr_dev_wer, self.step)
                for task in ['att', 'ctc']:
                    asr_dev_wer[task] = sum(asr_dev_wer[task])/len(asr_dev_wer[task])
                    if asr_dev_wer[task] < self.solver_asr.best_wer[task]:
                        self.solver_asr.best_wer[task] = asr_dev_wer[task]
                        self.save_checkpoint('asr_best_{}.pth'.format(task), {'wer': asr_dev_wer[task]})
                    self.write_log('wer', {'dv_'+task: asr_dev_wer[task]})
                metric_dict['wer'] = asr_dev_wer['att'] 

                asr_dev_wer_ctc = asr_dev_wer['ctc']
                asr_dev_wer_att = asr_dev_wer['att']
                print(f'[ASR] - CTC_WER = {asr_dev_wer_ctc}, ATT_WER = {asr_dev_wer_att}')

                if self.paras.save_spk_emb:
                    self.log.add_embedding(np.array(asr_spk_emb_list), asr_spk_list,  global_step=self.step, tag='asr')

        if self.se_task:
            print('[SE] - Logging metrics and media files.')
            self.solver_se.log_evaluation(step=self.step, split='dev', save_fn=self.save_checkpoint)
            print('[SE] - Logging finished.')

            if self.rank == 0:
                if self.paras.save_spk_emb:
                    self.log.add_embedding(np.array(se_spk_emb_list), se_spk_list,  global_step=self.step, tag='se')
        
        if self.sc_task:
            if self.solver_sc.rank == 0:
                print('[SC] - Logging metrics and media files.')
                self.sc_acc_list = self.sc_dev_tensor_list.cpu().numpy().tolist()
                ACC=self.solver_sc.calculate_metric(self.sc_acc_list)
                self.solver_sc.logging("sc_dev_acc", ACC, self.step)
                print(f'[SC] SC accuracy : {ACC*100} %')
                if self.solver_sc.best_acc['accuracy'] < ACC:
                    self.solver_sc.best_acc['accuracy'] = ACC
                    self.save_checkpoint('sc_best_dev.pth',{'sc_best_acc': ACC},show_msg=False)
                    print(f'[SC] ready to save SC model')
                else:
                    print(f'[SC] Model performance is not higher than the previous best one, so we skip it')
                print(f'[SC] Logging finished.')

                if self.paras.save_spk_emb:
                    self.log.add_embedding(np.array(sc_spk_emb_list), sc_spk_list,  global_step=self.step, tag='sc')

        if self.tts_task:
            if self.solver_tts.rank == 0:
                print('[TTS] - Logging started')
                tts_dev_avg_loss = sum(self.tts_dev_loss_list) / len(self.tts_dev_loss_list)
                tts_dev_avg_mel_loss = sum(self.tts_dev_mel_loss_list) / len(self.tts_dev_mel_loss_list)
                tts_dev_avg_d_loss = sum(self.tts_dev_d_loss_list) / len(self.tts_dev_d_loss_list)
                tts_dev_avg_speaker_loss = sum(self.tts_dev_speaker_loss_list) / len(self.tts_dev_speaker_loss_list)
                tts_dev_avg_prosody_loss = sum(self.tts_dev_prosody_loss_list) / len(self.tts_dev_prosody_loss_list)
                self.solver_tts.logging("tts_dev_loss", tts_dev_avg_loss, self.step, mode='scalar')
                self.solver_tts.logging("tts_dev_mel_loss", tts_dev_avg_mel_loss, self.step, mode='scalar')
                self.solver_tts.logging("tts_dev_d_loss", tts_dev_avg_d_loss, self.step, mode='scalar')
                self.solver_tts.logging("tts_dev_speaker_loss", tts_dev_avg_speaker_loss, self.step, mode='scalar')
                self.solver_tts.logging("tts_dev_prosody_loss", tts_dev_avg_prosody_loss, self.step, mode='scalar')
                for idx in range(len(self.tts_sample_mel_list)):
                    tts_dev_sample_mel = self.tts_sample_mel_list[idx]
                    tts_dev_target_mel = self.tts_target_mel_list[idx]
                    tts_dev_sample_wav = self.tts_sample_wav_list[idx]
                    tts_dev_target_wav = self.tts_target_wav_list[idx]
                
                    self.solver_tts.logging(f"tts_sample_mel_{idx}", tts_dev_sample_mel, self.step, mode='image')
                    self.solver_tts.logging(f"tts_target_mel_{idx}", tts_dev_target_mel, self.step, mode='image')
                    self.solver_tts.logging(f"TTS_sample_wav_{idx}", tts_dev_sample_wav, self.step, mode='wav')
                    self.solver_tts.logging(f"TTS_target_wav_{idx}", tts_dev_target_wav, self.step, mode='wav')

                if tts_dev_avg_mel_loss < self.solver_tts.best_loss:
                    self.solver_tts.best_loss = tts_dev_avg_mel_loss
                    self.save_checkpoint('tts_best.pth', {'tts_best_loss':tts_dev_avg_mel_loss},show_msg=False)
                print(f'[TTS] - MEL_LOSS = {tts_dev_avg_mel_loss}, D_LOSS = {tts_dev_avg_d_loss}')
                print('[TTS] - Logging finished')

                if self.paras.save_spk_emb:
                    self.log.add_embedding(np.array(tts_spk_emb_list), tts_spk_list,  global_step=self.step, tag='tts')


        if self.vcb_task:
            if self.solver_vcb.rank == 0:
                print('[VCB] - Logging started')
                vcb_avg_loss = {}
                for i, task in enumerate(['reconstruction', 'conversion', 'prosody']):
                    vcb_avg_loss[task] = sum(self.vcb_dev_loss_list[i]) / len(self.vcb_dev_loss_list[i])
                    if vcb_avg_loss[task] < self.solver_vcb.best_losses[task]:
                        self.solver_vcb.best_losses[task] = vcb_avg_loss[task]
                        self.save_checkpoint('vcb_best_{}.pth'.format(task),{'vcb_{}'.format(task): vcb_avg_loss[task]},show_msg=False)
                    self.solver_vcb.log('vcb_dev_{}_loss'.format(task), vcb_avg_loss[task].item(), self.step)
                self.solver_vcb.log_evaluation(step=self.step, split='dev')
                metric_dict['vcb_convert'] = vcb_avg_loss['conversion']

                vcb_print = vcb_avg_loss['conversion']
                print(f'[VCB] - CONVERSION_LOSS = {vcb_print}')
                print('[VCB] - Logging finished')

                if self.paras.save_spk_emb:
                    self.log.add_embedding(np.array(vcb_spk_emb_list), vcb_spk_list,  global_step=self.step, tag='vcb')

        # Save ckpt at every step
        if self.rank == 0:
            self.save_checkpoint('latest.pth', metric_dict, show_msg=False)

        # Resume training
        self.model.train()
        if self.asr_task:
            if self.solver_asr.emb_decoder is not None:
                self.solver_asr.emb_decoder.train()
        print(f'Validation finished.')
